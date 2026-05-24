import socket
import torch
import json
import argparse
import os
import time
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import re

from transformers import AutoTokenizer, AutoConfig
from llava.mm_utils import KeywordsStoppingCriteria, process_image, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model


class VLMServer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.vision_tower = None
        self.setup()

    def setup(self):
        self._disable_initializers()
        self._initialize_tokenizer_and_model()
        
        if self.args.precision == "W16A16":
            self._load_checkpoint_w16a16()
        else:
            raise ValueError(f"Precision {self.args.precision} not supported")

    def _disable_initializers(self):
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None

    def _initialize_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.model_path, "llm"), use_fast=False
        )
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

    def _load_checkpoint_w16a16(self):
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for _ in pbar:
            # self.model.llm = load_checkpoint_and_dispatch(
            #     self.model.llm,
            #     os.path.join(self.args.model_path, "llm"),
            #     no_split_module_classes=[
            #         "OPTDecoderLayer",
            #         "LlamaDecoderLayer",
            #         "BloomBlock",
            #         "MPTBlock",
            #         "DecoderLayer",
            #         "CLIPEncoderLayer",
            #     ],
            # ).to(self.args.device)
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path, model_name, None,
                load_8bit=getattr(args, "load_8bit", False),
            )
            self.tokenizer =  tokenizer
            self.model = model
            self.image_processor = image_processor
        # int8 model is already placed by accelerate's device_map; .to() would error
        if not getattr(self.args, "load_8bit", False):
            self.model = self.model.to(self.args.device)

    def _recv_request(self, conn):
        """Read length-prefixed JSON payload from a connection."""
        size_data = conn.recv(8)
        if len(size_data) < 8:
            return None
        size = int.from_bytes(size_data, 'big')
        data = b''
        while len(data) < size:
            packet = conn.recv(min(65536, size - len(data)))
            if not packet:
                return None
            data += packet
        return json.loads(data.decode())

    def _send_response(self, conn, response):
        rb = json.dumps(response).encode()
        try:
            conn.sendall(len(rb).to_bytes(8, 'big'))
            conn.sendall(rb)
        except (BrokenPipeError, OSError):
            pass

    def start_server(self, host='localhost', port=12345):
        """Continuous-batching VLM server.

        Producer threads (one per accepted connection) read the request payload
        and put (request, response_event, response_slot, conn) on the queue.
        A single worker thread drains the queue every `batch_wait_ms` (or when
        `batch_cap` requests are pending) and runs ONE batched generate, then
        sends per-request responses back to each conn.

        Per-request items themselves can be batched (a multi-env request with
        N items counts as N rows in the batched generate). Multiple shards
        thus get merged into a single generate when their calls arrive close
        in time, eliminating idle gaps between shards.
        """
        import threading, queue
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(64)
        print(f"VLM Server listening on {host}:{port}  "
              f"batch_cap={self.args.batch_cap} batch_wait_ms={self.args.batch_wait_ms}", flush=True)

        req_q = queue.Queue()

        def producer(conn, addr):
            try:
                req = self._recv_request(conn)
                if req is None:
                    conn.close(); return
                # Normalize to batched form: a list of (images_b64, instruction)
                # tuples; single-request calls become a 1-element batch.
                if 'images_list' in req and 'queries' in req:
                    items = list(zip(req['images_list'], req['queries']))
                elif 'images' in req and 'query' in req:
                    items = [(req['images'], req['query'])]
                else:
                    self._send_response(conn, {"error": "bad request schema"})
                    conn.close(); return
                done_evt = threading.Event()
                slot = [None]
                req_q.put((items, done_evt, slot, conn))
                done_evt.wait(timeout=600)
                # Worker filled slot[0] with list[str] matching `items` order.
                if slot[0] is None:
                    self._send_response(conn, {"error": "worker timeout"})
                elif len(items) == 1 and 'query' in req:
                    self._send_response(conn, slot[0][0])  # single-API back-compat
                else:
                    self._send_response(conn, slot[0])
            except Exception as _e:
                try:
                    self._send_response(conn, {"error": str(_e)})
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        def worker():
            import time as _t
            while True:
                # Wait for first item, then drain up to batch_cap within wait window
                first = req_q.get()
                pending = [first]
                start = _t.time()
                while len(pending) < self.args.batch_cap:
                    remaining_ms = self.args.batch_wait_ms - (_t.time() - start) * 1000
                    # Also stop if accumulated rows already >= batch_cap
                    total_rows = sum(len(it[0]) for it in pending)
                    if total_rows >= self.args.batch_cap:
                        break
                    if remaining_ms <= 0:
                        break
                    try:
                        nxt = req_q.get(timeout=remaining_ms / 1000.0)
                        pending.append(nxt)
                    except queue.Empty:
                        break

                # Flatten: collect all items across pending requests
                all_items = []
                slices = []  # per-request (start_idx, n_items) so we can split results
                for items, _evt, _slot, _conn in pending:
                    slices.append((len(all_items), len(items)))
                    all_items.extend(items)
                images_list = [it[0] for it in all_items]
                queries = [it[1] for it in all_items]
                t_g = _t.time()
                try:
                    raws = self.process_request_batch(images_list, queries)
                except Exception as _e:
                    print(f"[worker] generate failed: {_e}", flush=True)
                    raws = [""] * len(all_items)
                dt = _t.time() - t_g
                print(f"[worker] N_rows={len(all_items)} from {len(pending)} requests  generate={dt:.2f}s", flush=True)
                # Dispatch back
                for (st, n), (_items, evt, slot, _conn) in zip(slices, pending):
                    slot[0] = raws[st:st + n]
                    evt.set()

        threading.Thread(target=worker, daemon=True, name="vlm-worker").start()

        while True:
            conn, addr = server_socket.accept()
            threading.Thread(target=producer, args=(conn, addr), daemon=True,
                             name=f"vlm-prod-{addr[1]}").start()

    def process_request_batch(self, images_list, queries):
        """Batched: N envs in one generate call. Returns list[str] of length N.

        Uses the FIXED batched-VLM path (BOS-prefill for short rows, attn=all-1)
        to dodge the prepare_inputs_labels_for_multimodal mixed-length bug
        (left/right pad with attn=0 mask makes row>=1 output garbage like '.').
        See /tmp/diag_batched_vlm.py for the diagnostic that established this.
        """
        N = len(images_list)
        assert len(queries) == N

        # Decode + process per-env images → list of [F, ...] float16 tensors
        per_env_tensors = []
        for imgs_b64 in images_list:
            pil_imgs = []
            for b64 in imgs_b64:
                try:
                    pil = Image.open(BytesIO(base64.b64decode(b64))).convert('RGB')
                except Exception:
                    pil = Image.new('RGB', (224, 224), (0, 0, 0))
                pil_imgs.append(pil)
            # pad to num_video_frames if short
            F = self.args.num_video_frames
            if len(pil_imgs) < F:
                pad = pil_imgs[-1] if pil_imgs else Image.new('RGB', (224, 224), (0, 0, 0))
                pil_imgs = [Image.new('RGB', pad.size, (0, 0, 0))] * (F - len(pil_imgs)) + pil_imgs
            pil_imgs = pil_imgs[-F:]  # last F frames
            self.model.config.image_processor = self.image_processor
            processed = [process_image(im, self.model.config, None) for im in pil_imgs]
            per_env_tensors.append(torch.stack(processed, dim=0).to(self.args.device, dtype=torch.float16))

        # Build per-env prompts and tokenize
        image_token = "<image>\n"
        tok_lists = []
        for q in queries:
            conv = conv_templates[self.args.conv_mode].copy()
            qs = (
                f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f'of historical observations {image_token * (self.args.num_video_frames - 1)}, and current observation <image>\n. '
                f'Your assigned task is: "{q}" '
                f"Analyze this series of images to decide your next action, which could be turning left or right "
                f"by a specific degree, moving forward a certain distance, or stop if the task is completed."
            )
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            tok_lists.append(ids)

        # CRITICAL FIX: BOS-prefill short rows to Lmax, attn=all-1 (no padding mask)
        Lmax = max(t.shape[0] for t in tok_lists)
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else (
            self.tokenizer.eos_token_id or 0)
        input_ids = torch.full((N, Lmax), bos_id, dtype=tok_lists[0].dtype)
        for i, t in enumerate(tok_lists):
            input_ids[i, Lmax - t.shape[0]:] = t
        attn = torch.ones((N, Lmax), dtype=torch.long)
        input_ids = input_ids.to(self.args.device)
        attn = attn.to(self.args.device)

        conv0 = conv_templates[self.args.conv_mode]
        stop_str = conv0.sep if conv0.sep_style != SeparatorStyle.TWO else conv0.sep2
        sc = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        with torch.inference_mode():
            t0 = time.time()
            out_ids = self.model.generate(
                input_ids, attention_mask=attn, images=per_env_tensors,
                do_sample=False, temperature=0, top_p=None, num_beams=1,
                max_new_tokens=32, use_cache=True, stopping_criteria=[sc],
            )
            print(f"[batched-gen N={N}] {time.time() - t0:.2f}s", flush=True)
        raws = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        out = []
        for r in raws:
            r = r.strip()
            p = r.find('.')
            if p >= 0:
                r = r[:p + 1]
            out.append(r)
        torch.cuda.empty_cache()
        return out

    def process_request(self, images, query):
        # Process images
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.args.device, dtype=torch.float16)

        # Prepare prompt
        conv = conv_templates[self.args.conv_mode].copy()
        instruction = query
        image_token = "<image>\n"
        qs = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f'of historical observations {image_token * (self.args.num_video_frames-1)}, and current observation <image>\n. Your assigned task is: "{instruction}" '
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward a certain distance, or stop if the task is completed."
        )
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Generate response
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.args.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                images=[image_tensor],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=32,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            generation_time = time.time() - start_time
            print(f"Model generation took {generation_time:.2f} seconds")
            # print("input_ids:", input_ids)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        result = outputs.strip()
        # release activation/KV fragmentation back to allocator — recovers ~20-50 MiB
        # which is what closes the 48 MiB headroom gap when Isaac shares this GPU.
        torch.cuda.empty_cache()
        return result


def process_images(images, image_processor, model_cfg):
    """Process a list of images (either PIL Images or base64 strings)."""
    model_cfg.image_processor = image_processor
    processed_images = []
    
    for image in images:
        if isinstance(image, str):
            # Handle base64 encoded image
            try:
                # Decode base64 string to PIL Image
                image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
                # Create a blank image if decoding fails
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Process the PIL Image
        processed_image = process_image(image, model_cfg, None)
        processed_images.append(processed_image)

    if all(x.shape == processed_images[0].shape for x in processed_images):
        processed_images = torch.stack(processed_images, dim=0)
    return processed_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server")
    parser.add_argument("--port", type=int, default=54321, help="Port to bind the server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--precision", type=str, default="W16A16", help="compute precision")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--load_8bit", action="store_true", default=False,
                        help="Quantize VLM to int8 via bitsandbytes; ~10 GB vs ~17 GB fp16. "
                             "Use when co-locating with Isaac Sim on the same GPU.")
    parser.add_argument("--batch_cap", type=int, default=8,
                        help="Max total rows in one batched generate (continuous batching).")
    parser.add_argument("--batch_wait_ms", type=int, default=50,
                        help="Wait this many ms after first request to accumulate concurrent "
                             "requests before running generate.")
    args = parser.parse_args()
    
    server = VLMServer(args)
    server.start_server(host=args.host, port=args.port)
