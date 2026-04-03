"""VimRAG: Visual Memory RAG Agent - Demo Version"""

import json
import json5
import base64
import math
import requests
import torch
import os

from io import BytesIO
from PIL import Image
from torchcodec.decoders import VideoDecoder

from vimrag_prompt import SYSTEM_PROMPT, USER_PROMPT, SUMMARY_PROMPT
from vimrag_utils import fast_process_messages, calculate_intuitive_memory_energy, generate_multimodal_messages


class VimRAG:
    """VimRAG: Visual Memory RAG Agent with DAG-based reasoning."""
    
    def __init__(self, 
                 base_url, 
                 search_url, 
                 model_name,
                 api_key="EMPTY",
                 max_pixels_image=1003520,
                 max_pixels_video=401408,
                 memory_buffer_pixels=2007040,
                 search_top_k=3,
                 max_mem_steps=20,
                 enable_thinking=True):
        """
        Initialize VimRAG agent.
        
        Args:
            base_url: vLLM service base URL (e.g., http://localhost:8000/v1)
            search_url: Search service URL (e.g., http://localhost:8001/search)
            model_name: Model name for vLLM API
            api_key: API key for vLLM service authentication (default: "EMPTY")
            max_pixels_image: Max pixels per image
            max_pixels_video: Max pixels per video frame  
            memory_buffer_pixels: Total pixel budget for memory buffer
            search_top_k: Number of top results from search
            max_mem_steps: Maximum reasoning steps
            enable_thinking: Enable Qwen3.5 thinking mode (default: True)
        """
        self.base_url = base_url.rstrip('/')
        self.search_url = search_url
        self.model_name = model_name
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        
        self.max_pixels_image = max_pixels_image
        self.max_pixels_video = max_pixels_video
        self.memory_buffer_pixels = memory_buffer_pixels
        self.search_top_k = search_top_k
        self.max_mem_steps = max_mem_steps
        
        # Video processing settings
        self.fps_video = 1.0
        self.max_frames_video = 32

    # ==================== API Methods ====================
    
    def _model_generate(self, messages):
        """Call vLLM OpenAI-compatible API with streaming."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        # 思考模式用较高温度，非思考用 0
        if self.enable_thinking:
            payload["temperature"] = 0.6
            payload["top_p"] = 0.95
        else:
            payload["temperature"] = 0
        
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
        
        reasoning_content = ""
        content = ""
        
        for line in response.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]
            if line.strip() == '[DONE]':
                break
            try:
                chunk = json.loads(line)
                delta = chunk['choices'][0]['delta']
                # 思考内容在 reasoning_content 字段 (有些版本叫 reasoning)
                reasoning_piece = delta.get('reasoning_content') or delta.get('reasoning') or ''
                content_piece = delta.get('content') or ''
                
                if reasoning_piece:
                    reasoning_content += reasoning_piece
                    yield {"type": "think", "content": reasoning_piece}
                if content_piece:
                    content += content_piece
                    yield {"type": "content", "content": content_piece}
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        
        # 最终完整结果
        yield {"type": "done", "reasoning": reasoning_content, "content": content}
    
    def search(self, queries, top_k=None):
        """Call search service API."""
        if isinstance(queries, str):
            queries = [queries]
        top_k = top_k or self.search_top_k
        response = requests.post(self.search_url, json={"queries": queries, "top_k": top_k})
        response.raise_for_status()
        return response.json()['results']
    
    def format_search_results(self, search_results, add_vision_ids=False):
        """Format search results to OpenAI message format."""
        video_count, image_count, text_count = 0, 0, 0
        vision_ids_dict = {}
        
        user_content = [{"type": "text", "text": "<tool_response>\nHere are the retrieved information:"}]
        
        for modality in ['text', 'image', 'video']:
            for item in search_results.get('data', []):
                if item['type'] != modality:
                    continue
                    
                if item['type'] == 'image':
                    image_count += 1
                    if add_vision_ids:
                        user_content.append({"type": "text", "text": f"Picture {image_count}:"})
                        vision_ids_dict[f"Picture {image_count}"] = item['file_path']
                    user_content.append({
                        "type": "image",
                        "image": item['file_path'],
                        "max_pixels": self.max_pixels_image
                    })
                    
                elif item['type'] == 'video':
                    video_count += 1
                    if add_vision_ids:
                        user_content.append({"type": "text", "text": f"Video {video_count}:"})
                        vision_ids_dict[f"Video {video_count}"] = item['file_path']
                    user_content.append({
                        "type": "video",
                        "video": item['file_path'],
                        "max_pixels": self.max_pixels_video,
                        "fps": self.fps_video,
                        "max_frames": self.max_frames_video
                    })
                    
                elif item['type'] == 'text':
                    text_count += 1
                    if add_vision_ids:
                        user_content.append({"type": "text", "text": f"Text {text_count}:"})
                        vision_ids_dict[f"Text {text_count}"] = item['content']
                    user_content.append({"type": "text", "text": item['content']})
        
        user_content.append({"type": "text", "text": "\n</tool_response>"})
        
        if add_vision_ids:
            return user_content, vision_ids_dict
        return user_content

    # ==================== Video Processing (In-Memory) ====================
    
    def _process_image_to_base64(self, image_input, max_pixels=1003520):
        """Process image to base64 string (in-memory, no disk I/O)."""
        MAX_BYTES = 38000
        
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB") if hasattr(image_input, 'convert') else image_input
        
        w, h = img.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            img = img.resize((int(w * scale), int(h * scale)), resample=Image.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=75)
        
        if buffered.tell() > MAX_BYTES:
            scale = math.sqrt((MAX_BYTES * 0.95) / buffered.tell())
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), resample=Image.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=75)
        
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _get_video_frames_base64(self, video_path, target_timestamps, fps=1.0, max_frames=32, max_pixels=401408):
        """Extract video frames as base64 strings (in-memory, no disk I/O)."""
        decoder = VideoDecoder(video_path, device="cpu")
        metadata = decoder.metadata
        duration = metadata.duration_seconds
        total_frames = metadata.num_frames
        
        needed_frames = max(1, int(duration * fps))
        num_samples = min(needed_frames, max_frames)
        indices = torch.linspace(0, total_frames - 1, steps=num_samples).long()
        
        # Build theoretical timestamp mapping
        ts_map = {}
        for i, idx in enumerate(indices):
            ts_str = f"{(idx.item() / total_frames) * duration:.1f}"
            ts_map[ts_str] = idx.item()
        
        # Extract frames for target timestamps
        result_base64 = []
        for ts in target_timestamps:
            ts_str = f"{float(ts):.1f}"
            if ts_str not in ts_map:
                continue
            
            frame_idx = ts_map[ts_str]
            frame_packet = decoder[frame_idx]
            img_tensor = frame_packet.data if hasattr(frame_packet, 'data') else frame_packet
            img = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())
            
            base64_str = self._process_image_to_base64(img, max_pixels)
            result_base64.append(f"data:image/jpeg;base64,{base64_str}")
        
        return result_base64

    # ==================== Core Logic ====================

    def _build_initial_messages(self, question, action_graph):
        """Build initial messages for the model."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT.format(
                    query=question, 
                    action_graph=json.dumps(action_graph, ensure_ascii=False)
                )}
            ]}
        ]

    def _update_messages_with_memory(self, messages, action_graph, multimodal_memory):
        """Update messages with multimodal memory content."""
        if not multimodal_memory:
            return
        
        final_mem_energies = calculate_intuitive_memory_energy(action_graph, multimodal_memory)
        openai_messages = generate_multimodal_messages(
            action_graph, final_mem_energies, top_k=5,
            S_total=self.memory_buffer_pixels, 
            video_frame_max_pixels=self.max_pixels_video, 
            image_max_pixels=self.max_pixels_image
        )
        messages[-1]['content'].append({
            "type": "text", 
            "text": "### Multimodal Memory\nHistorical information for reference only.\n"
        })
        messages[-1]['content'].extend(openai_messages)

    def _parse_response(self, response):
        """Parse model response content and extract tool call."""
        if "<tool_call>" not in response or "</tool_call>" not in response:
            return None, None
        try:
            tool_json = json5.loads(response.split("<tool_call>")[1].split("</tool_call>")[0])
            return tool_json.get('name'), tool_json.get('arguments')
        except Exception as e:
            print(f"Parse error: {e}")
            return None, None

    def _handle_search_node(self, messages, response, tool_params):
        """Handle add_search_node tool call."""
        query = tool_params.get('query')
        search_results = self.search([query], top_k=self.search_top_k)[0]
        content, vision_ids = self.format_search_results(search_results, add_vision_ids=True)
        
        messages.extend([
            {"role": "assistant", "content": response},
            {"role": "user", "content": content},
            {"role": "user", "content": SUMMARY_PROMPT}
        ])
        return search_results, vision_ids

    def _handle_summarize_memorize(self, tool_params, vision_ids_dict, multimodal_memory, last_graph_node):
        """Handle summarize_and_memorize tool call (in-memory processing)."""
        memorize = tool_params.get('memorize', [])
        last_graph_node['summary'] = tool_params.get('summarize')
        node_memory = []
        
        for item in memorize:
            info_id = item['information_id']
            priority = item['priority_score']
            
            if not item.get('is_useful', False):
                continue
            
            if 'Video' in info_id:
                # Process video frames to base64 (no disk I/O)
                timestamps = item.get('key_timestamp', [])
                video_path = vision_ids_dict.get(info_id)
                if video_path and timestamps:
                    frames_base64 = self._get_video_frames_base64(
                        video_path, timestamps,
                        fps=self.fps_video, max_frames=self.max_frames_video,
                        max_pixels=self.max_pixels_video
                    )
                    if frames_base64:
                        node_memory.append({
                            'type': 'video',
                            'file_path': frames_base64,  # base64 URLs instead of file paths
                            'priority_score': priority
                        })
                        
            elif 'Picture' in info_id:
                image_path = vision_ids_dict.get(info_id)
                if image_path:
                    # Convert image to base64 (no disk I/O)
                    base64_str = self._process_image_to_base64(image_path, self.max_pixels_image)
                    node_memory.append({
                        'type': 'image',
                        'file_path': f"data:image/jpeg;base64,{base64_str}",
                        'priority_score': priority
                    })
        
        if node_memory:
            multimodal_memory[last_graph_node['id']] = node_memory

    def run(self, sample):
        """
        Run the VimRAG agent on a sample query. Yields progress events in real-time.
        
        Args:
            sample: Dict with 'query' key
            
        Yields:
            Dict with 'event' key and event-specific data:
            - {"event": "think", "content": str} - Thinking content chunk
            - {"event": "content", "content": str} - Output content chunk
            - {"event": "search", "query": str} - Search initiated
            - {"event": "search_done", "results": dict} - Search completed
            - {"event": "memorize", "summary": str} - Memorize action
            - {"event": "answer", "content": str, "sample": dict} - Final answer
            - {"event": "error", "content": str} - Error occurred
            - {"event": "max_steps", "content": str} - Maximum steps reached
        """
        question = sample['query']
        trajectory = []
        search_results_list = []
        
        # Initialize action graph with root node
        action_graph = [{
            "id": "root",
            "name": "Initial Node",
            "content": f"Initial query from user: {question}"
        }]
        multimodal_memory = {}
        
        need_update_context = True
        can_search = True
        last_graph_node = None
        vision_ids_dict = None
        generate_times = 0
        steps_remaining = self.max_mem_steps

        while steps_remaining > 0:
            steps_remaining -= 1
            
            # Build or update context
            if need_update_context:
                messages = self._build_initial_messages(question, action_graph)
                self._update_messages_with_memory(messages, action_graph, multimodal_memory)
                need_update_context = False
                can_search = True

            try:
                # Generate model response with streaming
                messages_base64 = fast_process_messages(messages)
                
                # 流式收集模型输出
                full_response = ""
                full_reasoning = ""
                for chunk in self._model_generate(messages_base64):
                    if chunk["type"] == "think":
                        yield {"event": "think", "content": chunk["content"]}
                    elif chunk["type"] == "content":
                        yield {"event": "content", "content": chunk["content"]}
                    elif chunk["type"] == "done":
                        full_response = chunk["content"]
                        full_reasoning = chunk["reasoning"]
                
                # Parse response
                tool_name, tool_params = self._parse_response(full_response)
                if tool_name is None:
                    yield {"event": "error", "content": "Failed to parse response"}
                    continue

                # Handle tool calls
                if tool_name == 'add_search_node' and can_search:
                    yield {"event": "search", "query": tool_params.get('query'), "node_id": tool_params.get('id', '')}
                    results, vision_ids_dict = self._handle_search_node(messages, full_response, tool_params)
                    last_graph_node = tool_params
                    search_results_list.append(results)
                    yield {"event": "search_done", "results": results}
                    can_search = False

                elif tool_name == 'summarize_and_memorize':
                    yield {"event": "memorize", "summary": tool_params.get('summarize')}
                    self._handle_summarize_memorize(tool_params, vision_ids_dict, multimodal_memory, last_graph_node)
                    messages.append({"role": "assistant", "content": full_response})
                    action_graph.append(last_graph_node.copy())
                    trajectory.append(messages.copy())
                    need_update_context = True

                elif tool_name == 'add_answer_node':
                    messages.append({"role": "assistant", "content": full_response})
                    trajectory.append(messages.copy())
                    
                    # 将 answer 节点加入 action_graph
                    action_graph.append({
                        "id": "answer",
                        "parent_ids": tool_params.get('parent_ids', []),
                    })
                    
                    sample['trajectory'] = trajectory
                    sample['search_results'] = search_results_list
                    sample['graph'] = action_graph
                    sample['generate'] = tool_params['answer']
                    sample['generate_times'] = generate_times
                    yield {"event": "answer", "content": tool_params['answer'], "sample": sample}
                    return
                else:
                    raise Exception(f"Unknown tool: {tool_name}")
                    
            except KeyboardInterrupt:
                exit()
            except Exception as e:
                yield {"event": "error", "content": str(e)}
                continue
            
            generate_times += 1
        
        yield {"event": "max_steps", "content": "Reached maximum steps"}


if __name__ == '__main__':
    # Example usage with streaming output
    agent = VimRAG(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        search_url="http://localhost:8001/search",
        model_name="qwen3.6-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        enable_thinking=False
    )
    
    # 流式输出示例：直接打印原始模型输出
    sample = {'query': '阿里云上数据中心和安全云服务分别说的什么?'}
    graph = None
    for event in agent.run(sample):
        if event['event'] in ('think', 'content'):
            print(event['content'], end='', flush=True)
        elif event['event'] == 'answer':
            graph = event['sample'].get('graph', [])

    # 在终端打印 DAG 图形
    if graph:
        print('\n')
        print('┌─────────────────────────────────┐')
        print('│         Action Graph            │')
        print('└─────────────────────────────────┘')
        # 构建子节点映射
        children = {}
        for node in graph:
            nid = node.get('id', '')
            children.setdefault(nid, [])
            for pid in node.get('parent_ids', []):
                children.setdefault(pid, [])
                children[pid].append(nid)

        icons = {'root': '◉', 'answer': '✦'}
        default_icon = '◆'

        def print_tree(nid, prefix='', is_last=True):
            icon = icons.get(nid, default_icon)
            if not prefix:
                print(f'  {icon} {nid}')
            else:
                connector = '└── ' if is_last else '├── '
                print(f'  {prefix}{connector}{icon} {nid}')
            child_list = children.get(nid, [])
            for i, cid in enumerate(child_list):
                ext = '    ' if is_last else '│   '
                print_tree(cid, prefix + ext, i == len(child_list) - 1)

        print_tree('root')

