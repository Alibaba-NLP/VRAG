import base64
import math
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from torchcodec.decoders import VideoDecoder


def process_image_to_base64(image_input, max_pixels=1004416):
    """
    处理图片并转换为 base64 字符串。
    确保图片体积不超过 38KB，同时尊重 max_pixels 限制。
    
    Returns: (base64_str, img, actual_total_pixels)
    """
    MAX_BYTES = 38000
    
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input

    # 1. 第一次缩放：基于像素数限制
    w, h = img.size
    current_pixels = w * h
    if current_pixels > max_pixels:
        scale_factor = math.sqrt(max_pixels / current_pixels)
        img = img.resize((int(w * scale_factor), int(h * scale_factor)), resample=Image.LANCZOS)
    
    # 2. 第一次尝试性保存
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=75)
    
    # 3. 第二次缩放：基于体积限制
    current_size = buffered.tell()
    if current_size > MAX_BYTES:
        size_ratio = (MAX_BYTES * 0.95) / current_size
        scale_factor = math.sqrt(size_ratio)
        
        new_w, new_h = int(img.size[0] * scale_factor), int(img.size[1] * scale_factor)
        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=75)

    # 4. 最终输出处理
    actual_total_pixels = img.size[0] * img.size[1]
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return base64_str, img, actual_total_pixels


def process_single_video(video_path, fps=1.0, max_frames=32, max_pixels=1004416):
    """
    处理单个视频，提取帧并转换为 base64 格式的 OpenAI 消息内容。
    
    Returns: base64_content 列表
    """
    decoder = VideoDecoder(video_path, device="cpu")
    metadata = decoder.metadata
    duration = metadata.duration_seconds
    total_frames = metadata.num_frames
    
    needed_frames = max(1, int(duration * fps))
    num_samples = min(needed_frames, max_frames)
    indices = torch.linspace(0, total_frames - 1, steps=num_samples).long()
    
    base64_content = []

    for idx in indices:
        idx_val = idx.item()
        frame_packet = decoder[idx_val]
        img_tensor = frame_packet.data if hasattr(frame_packet, 'data') else frame_packet
        img = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())
        
        base64_str, _, _ = process_image_to_base64(img, max_pixels)
        
        current_ts = (idx_val / total_frames) * duration
        ts_text = f"<{current_ts:.1f} seconds>"

        base64_content.append({"type": "text", "text": ts_text})
        base64_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
            "max_pixels": max_pixels
        })
        
    return base64_content


def fast_process_messages(messages, fps=1.0, max_frames=32, max_pixels=1004416):
    """
    处理消息列表，将视频和图片转换为 base64 格式。
    
    Returns: 处理后的消息列表（base64 格式）
    """
    new_messages = []
    
    for msg in messages:
        res_msg = {"role": msg["role"]}
        raw_content = msg.get("content", "")
        
        if isinstance(raw_content, list):
            new_content = []
            for item in raw_content:
                if item.get("type") == "video":
                    v_path = item.get("video")
                    v_fps = item.get("fps", fps)
                    v_max_f = item.get("max_frames", max_frames)
                    v_max_p = item.get("max_pixels", max_pixels)
                    
                    try:
                        b64_list = process_single_video(v_path, v_fps, v_max_f, v_max_p)
                        new_content.extend(b64_list)
                    except Exception as e:
                        print(f"Error processing video {v_path}: {e}")
                        continue

                elif item.get("type") == "image":
                    img_path = item.get("image")
                    max_pixels_tmp = item.get("max_pixels", max_pixels)
                    
                    # 如果已经是 data URL，直接透传
                    if isinstance(img_path, str) and img_path.startswith("data:image/"):
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": img_path},
                            "max_pixels": max_pixels_tmp
                        })
                        continue
                    
                    # 从本地路径读图再转 base64
                    base64_str, _, _ = process_image_to_base64(img_path, max_pixels_tmp)
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
                        "max_pixels": max_pixels_tmp
                    })
                else:
                    new_content.append(item)
            
            # 如果都是 text，合并为字符串
            if all(item.get("type") == "text" for item in new_content):
                new_content = ''.join(item.get("text", "") for item in new_content)

            res_msg["content"] = new_content
        else:
            res_msg["content"] = raw_content
            
        new_messages.append(res_msg)
        
    return new_messages


def calculate_intuitive_memory_energy(action_graph, multimodal_memory, lambda_decay=0.1, gamma_gain=0.3):
    """
    计算多模态记忆的能量值（直观版本）。
    
    1. Phi_i 简化为出度（该动作引导了多少后续动作）
    2. Gamma 简化为子节点能量的直接反馈
    """
    node_ids = [node['id'] for node in action_graph]
    t_max = len(node_ids) - 1
    
    # 构建简单的邻接表
    adj = {node_id: [] for node_id in node_ids}
    out_degree = {node_id: 0 for node_id in node_ids}
    for node in action_graph:
        for p_id in node.get('parent_ids', []):
            if p_id in adj:
                adj[p_id].append(node['id'])
                out_degree[p_id] += 1

    # 1. 计算初始衰减能量 (Intrinsic Energy)
    intrinsic_energies = {}
    for i, node_id in enumerate(node_ids):
        phi_i = 1 + out_degree[node_id]  # 越是基石节点，Phi越高
        delta_t = t_max - i
        
        items = multimodal_memory.get(node_id, [])
        node_item_energies = []
        for item in items:
            p_im = item['priority_score'] / 5.0
            e_base = (p_im * phi_i) * np.exp(-lambda_decay * delta_t)
            node_item_energies.append(e_base)
        intrinsic_energies[node_id] = node_item_energies

    # 2. 计算回想强化 (Back-propagation Reinforcement)
    final_omegas = {node_id: list(intrinsic_energies[node_id]) for node_id in node_ids}
    
    for node_id in reversed(node_ids):
        children = adj[node_id]
        if not children:
            continue
        
        child_feedback = 0
        for c_id in children:
            if final_omegas[c_id]:
                child_feedback += np.mean(final_omegas[c_id])
        
        for m_idx in range(len(final_omegas[node_id])):
            final_omegas[node_id][m_idx] += gamma_gain * child_feedback

    # 3. 封装返回
    result = {}
    for node_id in node_ids:
        items = multimodal_memory.get(node_id, [])
        result[node_id] = [
            {**item, 'omega': final_omegas[node_id][idx]} 
            for idx, item in enumerate(items)
        ]
    return result


def generate_multimodal_messages(action_graph, final_mem_energies, S_total=1048576, video_frame_max_pixels=256*32*32, image_max_pixels=512*32*32, top_k=3):
    """
    生成多模态消息内容。
    
    1. 以 Item (整个视频或图片) 为单位进行 Top-K 筛选
    2. 保留视频的所有帧
    3. 视频内的所有帧共享该视频分到的总像素额度
    """
    # 1. 汇总所有 Item，计算它们的实体总能量
    all_items_list = []
    for node_id, items in final_mem_energies.items():
        for item in items:
            all_items_list.append({**item, 'node_id': node_id})

    # 2. Top-K 筛选 (以 Item 为单位)
    all_items_list.sort(key=lambda x: x['omega'], reverse=True)
    selected_items = all_items_list[:top_k]
    
    # 3. 计算选定 Item 的能量总和用于像素归一化
    total_omega_sum = sum(item['omega'] for item in selected_items) if selected_items else 1.0

    # 建立查找索引
    filtered_lookup = {node['id']: [] for node in action_graph}
    for item in selected_items:
        filtered_lookup[item['node_id']].append(item)

    content_list = []
    for node in action_graph:
        node_id = node['id']
        if node_id == 'root':
            continue
        
        items = filtered_lookup.get(node_id, [])
        if not items:
            continue

        content_list.append({"type": "text", "text": f"Node: {node_id}"})
        
        for item in items:
            item_total_pixels = int(S_total * (item['omega'] / total_omega_sum))
            
            frames = item['file_path'] if isinstance(item['file_path'], list) else [item['file_path']]
            num_frames = len(frames)
            
            pixels_per_frame = item_total_pixels // num_frames if num_frames > 0 else 0
            
            for frame_path in frames:
                content_list.append({
                    "type": "image",
                    "image": frame_path,
                    "max_pixels": pixels_per_frame
                })
        
    return content_list
