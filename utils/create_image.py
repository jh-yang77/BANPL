import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn.functional as F
import numpy as np

def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    """
    RN50 版本: 使用 Grad-CAM 生成 Visual Heatmap
    Text 版本: 保持原有的 Attention Rollout (因为文本端永远是 Transformer)
    """
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)

    # =============================================================
    # [新增] 1. 注册 Hooks (为了获取 RN50 layer4 的特征图和梯度)
    # =============================================================
    visual_grads = []
    visual_fmaps = []

    def save_fmap(module, input, output):
        visual_fmaps.append(output)

    def save_grad(module, grad_in, grad_out):
        visual_grads.append(grad_out[0])

    # CLIP 的 RN50 最后一层通常是 layer4
    # 如果是 ModifiedResNet (Dassl/OpenAI CLIP)，路径是 model.visual.layer4
    target_layer = model.visual.layer4
    
    handle_f = target_layer.register_forward_hook(save_fmap)
   # 新写法：加上 _full_
    handle_b = target_layer.register_full_backward_hook(save_grad)

    # =============================================================
    # 2. Forward Pass
    # =============================================================
    # 这一步会触发 save_fmap，记录下特征图
    logits_per_image, logits_per_text = model(images, texts)
    
    # 准备反向传播的目标 (One-hot)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    
    # 计算 Loss
    one_hot_score = torch.sum(one_hot * logits_per_image)
    
    # =============================================================
    # 3. Backward Pass
    # =============================================================
    model.zero_grad()
    # 这一步会触发 save_grad，记录下梯度
    one_hot_score.backward(retain_graph=True)

    # =============================================================
    # [修改] 4. 计算 Image Relevance (Grad-CAM)
    # =============================================================
    fmaps = visual_fmaps[0] # [B, 2048, 7, 7] (对于 RN50)
    grads = visual_grads[0] # [B, 2048, 7, 7]

    # (1) Global Average Pooling 获取权重
    weights = torch.mean(grads, dim=(2, 3), keepdim=True) # [B, 2048, 1, 1]
    
    # (2) 加权求和
    cam = torch.sum(weights * fmaps, dim=1) # [B, 7, 7]
    
    # (3) ReLU (只关注正向贡献)
    cam = F.relu(cam)
    
    # (4) 插值回 224x224
    # 注意：这里直接变成 [B, 224, 224]，你需要配合修改 coop.py 里的 resize 逻辑
    image_relevance = cam.unsqueeze(1) # [B, 1, 7, 7]
    image_relevance = F.interpolate(image_relevance, size=(224, 224), mode='bilinear', align_corners=False)
    image_relevance = image_relevance.squeeze(1) # [B, 224, 224]

    # 清除 Hooks，防止内存泄漏
    handle_f.remove()
    handle_b.remove()

    # =============================================================
    # 5. Text Relevance (保持原样，因为 Text Encoder 是 Transformer)
    # =============================================================
    # 注意：有些版本的 CLIP text transformer 路径可能略有不同
    # 如果 model.transformer 报错，尝试 model.text_encoder 等
    try:
        text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

        if start_layer_text == -1:
            start_layer_text = len(text_attn_blocks) - 1

        num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
        R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
        R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        
        for i, blk in enumerate(text_attn_blocks):
            if i < start_layer_text:
                continue
            # 注意：这里的 attn_probs 需要你在 Text Encoder 里也改过代码才能拿到
            # 如果没改过 CLIP 源码，这里可能会报错 'AttributeError'
            # 如果报错，建议直接返回 zeros
            grad = torch.autograd.grad(one_hot_score, [blk.attn_probs], retain_graph=True)[0].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        text_relevance = R_text
    except Exception as e:
        # 如果不需要文本解释，直接给个占位符
        # print(f"Text interpretation failed: {e}, using zeros.")
        text_relevance = torch.zeros((batch_size, 77, 77)).to(device)

    return text_relevance, image_relevance


def interpret_batch_paired(image, texts, model, device, start_layer=-1):
    """Compute CAM for paired batches: image[i] is explained by texts[i]."""
    batch_size = image.shape[0]
    if texts.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: {batch_size} images but {texts.shape[0]} texts")

    model.zero_grad(set_to_none=True)
    visual_grads = []
    visual_fmaps = []
    handle_f = None
    handle_b = None

    def save_fmap(module, input, output):
        visual_fmaps.append(output)

    def save_grad(module, grad_in, grad_out):
        visual_grads.append(grad_out[0])

    is_resnet = hasattr(model.visual, "layer4")
    if is_resnet:
        target_layer = model.visual.layer4
        handle_f = target_layer.register_forward_hook(save_fmap)
        handle_b = target_layer.register_full_backward_hook(save_grad)

    try:
        logits_per_image, _ = model(image, texts)
        target = torch.eye(batch_size, logits_per_image.shape[1], device=device, dtype=logits_per_image.dtype)
        one_hot_score = torch.sum(target * logits_per_image)
        one_hot_score.backward(retain_graph=True)

        if is_resnet:
            fmaps = visual_fmaps[0]
            grads = visual_grads[0]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * fmaps, dim=1)
            cam = F.relu(cam)
            image_relevance = F.interpolate(
                cam.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
            if start_layer == -1:
                start_layer = len(image_attn_blocks) - 1

            num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
            R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
            R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

            for i, blk in enumerate(image_attn_blocks):
                if i < start_layer:
                    continue
                grad = torch.autograd.grad(one_hot_score, [blk.attn_probs], retain_graph=True)[0].detach()
                cam = blk.attn_probs.detach()
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
                cam = cam.clamp(min=0).mean(dim=1)
                R = R + torch.bmm(cam, R)
            image_relevance = R[:, 0, 1:]
    finally:
        if handle_f is not None:
            handle_f.remove()
        if handle_b is not None:
            handle_b.remove()

    text_relevance = torch.zeros((batch_size, 77, 77), device=device)
    return text_relevance, image_relevance


def interpret_batch_attention_rollout(image, texts, model, device, start_layer=-1):
    """Return class-agnostic visual attention rollout for a paired image batch."""
    batch_size = image.shape[0]
    if texts.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: {batch_size} images but {texts.shape[0]} texts")
    if hasattr(model.visual, "layer4"):
        raise ValueError("Attention rollout is only defined for CLIP ViT backbones")

    model(image, texts)
    blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = len(blocks) - 1

    num_tokens = blocks[0].attn_probs.shape[-1]
    # Accumulate rollout in FP32. Multiplying all transformer layers in the
    # AMP attention dtype otherwise underflows many small transition weights.
    rollout = torch.eye(num_tokens, dtype=torch.float32, device=device)
    rollout = rollout.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for layer_idx, block in enumerate(blocks):
        if layer_idx < start_layer:
            continue
        attention = block.attn_probs.detach().float()
        attention = attention.reshape(batch_size, -1, num_tokens, num_tokens).mean(dim=1)
        attention = attention.clamp(min=0)
        identity = torch.eye(num_tokens, dtype=torch.float32, device=device).unsqueeze(0)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attention, rollout)

    text_relevance = torch.zeros((batch_size, 77, 77), device=device)
    return text_relevance, rollout[:, 0, 1:]
# def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
#     batch_size = texts.shape[0]
#     images = image.repeat(batch_size, 1, 1, 1)
#     logits_per_image, logits_per_text = model(images, texts)
#     probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
#     index = [i for i in range(batch_size)]
#     one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
#     one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
#     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#     one_hot = torch.sum(one_hot.cuda() * logits_per_image)
#     model.zero_grad()

#     image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

#     if start_layer == -1:
#         # calculate index of last layer
#         start_layer = len(image_attn_blocks) - 1

#     num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
#     R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
#     R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
#     for i, blk in enumerate(image_attn_blocks):
#         if i < start_layer:
#             continue
#         grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
#         cam = blk.attn_probs.detach()
#         cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#         grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#         cam = grad * cam
#         cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
#         cam = cam.clamp(min=0).mean(dim=1)
#         R = R + torch.bmm(cam, R)
#     image_relevance = R[:, 0, 1:]

#     text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

#     if start_layer_text == -1:
#         # calculate index of last layer
#         start_layer_text = len(text_attn_blocks) - 1

#     num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
#     R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
#     R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
#     for i, blk in enumerate(text_attn_blocks):
#         if i < start_layer_text:
#             continue
#         grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
#         cam = blk.attn_probs.detach()
#         cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
#         grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
#         cam = grad * cam
#         cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
#         cam = cam.clamp(min=0).mean(dim=1)
#         R_text = R_text + torch.bmm(cam, R_text)
#     text_relevance = R_text

#     return text_relevance, image_relevance

class FewShotDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item.impath).convert('RGB')
        image = self.transform(image)
        return {
            "img": image,
            "label": item.label,
            "classname": item.classname,
            "impath": item.impath
        }

def interpret_ours(
    model,
    image,
    texts=None,
    device=None,
    text_embeddings=None,
    start_layer=-1,
    start_layer_text=-1,
):
    """
    interpret(): 支持自定义 text_embeddings，用于 BANPL/CoOp Prompt 可视化。
    如果传入 text_embeddings，则不再调用 model(images, texts)。
    """
    batch_size = text_embeddings.shape[0] if text_embeddings is not None else texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)

    # === 根据输入类型计算 logits ===
    if text_embeddings is not None:
        # 使用 BANPL 的 prompt 特征
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        logits_per_image = (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()
    else:
        # 使用 CLIP 原生 encode_text
        logits_per_image, logits_per_text = model(images, texts)

    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    # === Image side relevance ===
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    # === Text side relevance ===
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    if start_layer_text == -1:
        start_layer_text = len(text_attn_blocks) - 1
    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance


def interpret_batch_vit_gradcam(image, texts, model, device, variant="gradcam"):
    """Compute paired Grad-CAM or Grad-CAM++ maps for a CLIP ViT batch."""
    batch_size = image.shape[0]
    if texts.shape[0] != batch_size:
        raise ValueError(f"Batch mismatch: {batch_size} images but {texts.shape[0]} texts")
    if hasattr(model.visual, "layer4"):
        raise ValueError("ViT Grad-CAM controls require a CLIP ViT backbone")
    if variant not in {"gradcam", "gradcampp"}:
        raise ValueError(f"Unsupported ViT CAM variant: {variant}")

    activations = []
    gradients = []
    target_layer = model.visual.transformer.resblocks[-1].ln_1

    def save_activation(module, inputs, output):
        activations.append(output)

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    model.zero_grad(set_to_none=True)

    try:
        logits_per_image, _ = model(image, texts)
        paired_score = logits_per_image.diagonal().sum()
        paired_score.backward(retain_graph=False)
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if len(activations) != 1 or len(gradients) != 1:
        raise RuntimeError(
            f"Expected one activation/gradient tensor, got {len(activations)}/{len(gradients)}"
        )

    activation = activations[0].detach().float()
    gradient = gradients[0].detach().float()
    if activation.ndim != 3:
        raise RuntimeError(f"Unexpected ViT activation shape: {tuple(activation.shape)}")

    # OpenAI CLIP transformer blocks use [tokens, batch, channels].
    activation = activation.permute(1, 2, 0)[:, :, 1:]
    gradient = gradient.permute(1, 2, 0)[:, :, 1:]
    patch_count = activation.shape[-1]
    side = int(round(patch_count ** 0.5))
    if side * side != patch_count:
        raise RuntimeError(f"Expected a square ViT patch grid, got {patch_count} tokens")
    activation = activation.reshape(batch_size, activation.shape[1], side, side)
    gradient = gradient.reshape(batch_size, gradient.shape[1], side, side)

    if variant == "gradcam":
        weights = gradient.mean(dim=(2, 3), keepdim=True)
    else:
        grad2 = gradient.pow(2)
        grad3 = grad2 * gradient
        activation_sum = activation.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * grad2 + activation_sum * grad3
        denominator = torch.where(
            denominator.abs() > 1e-7,
            denominator,
            torch.ones_like(denominator),
        )
        alpha = grad2 / denominator
        weights = (alpha * gradient.clamp(min=0)).sum(dim=(2, 3), keepdim=True)

    image_relevance = (weights * activation).sum(dim=1).clamp(min=0)
    image_relevance = image_relevance.flatten(1)
    text_relevance = torch.zeros((batch_size, 77, 77), device=device)
    return text_relevance, image_relevance
