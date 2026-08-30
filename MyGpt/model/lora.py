import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self, in_feature:int , out_feature: int, rank: int):
        super().__init__()
        self.rank = rank
        self.linear_a = nn.Linear(in_feature, rank, bias=False)
        self.linear_b = nn.Linear(rank, out_feature, bias=False)
        self.linear_a.weight.data.normal_(mean=0.0, std=0.02)
        self.linear_b.weight.data.zero_()

    def forward(self, x):
        return self.linear_b(self.linear_a(x))

def apply_lora_to_model(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank).to(next(module.parameters()).device)
            setattr(module, "lora", lora)
            origin_forward = module.forward

            def mixed_forward(x, forward1 = origin_forward, forward2 = lora):
                return forward1(x) + forward2(x)

            module.forward = mixed_forward

def save_lora(model, lora_path):
    # _orig_mod 是什么
    # torch.compile(model) 的返回值不是原来的模型，而是一个包装器 OptimizedModule，原始模型被存进它的 _orig_mod 属性：
    # model = torch.compile(model)          # model 现在是 OptimizedModule
    # model._orig_mod                        # ← 真正的原始模型
    raw_model = getattr(model, "_orig_mod", model)
    save_lora_state = {}

    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            real_name = name[7:] if name.startswith("module.") else name
            # named_parameters() 拿到 linear_a.weight / linear_b.weight 等张量
            lora_update_dict = {f"{real_name}.lora.{k}": v for k, v in module.lora.named_parameters()}
            save_lora_state.update(lora_update_dict)

    # 在这个函数里面只是把所有的lora权重存起来，并不保存原来的权重
    torch.save(save_lora_state, lora_path)

def load_lora(model, lora_path):
    raw_model = getattr(model, "_orig_mod", model)
    saved_state_dict = torch.load(lora_path, map_location=next(raw_model.parameters()).device)
    saved_state_dict = {(k[7:] if k.startswith("module.") else k) : v for k, v in saved_state_dict.items()}

    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            real_name = name[7:] if name.startswith("module.") else name
            lora_prefix = f"{real_name}.lora."
            # 这个地方的k和prefix要跟上面的 save_lora 函数对应
            lora_state_dict = { k.replace(lora_prefix, ''): v.cpu().half() for k, v in saved_state_dict.items() if lora_prefix in k }
            module.lora.load_state_dict(lora_state_dict)

def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = { k: v.cpu().half() for k, v in raw_model.state_dict().items() if ".lora." not in k }
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            state_dict[f'{name}.weight'] = module.weight.data.cpu().half()
            state_dict[f'{name}.weight'] += (module.lora.linear_b.weight.data @ module.lora.linear_a.weight.data).cpu().half()
    torch.save(state_dict, save_path)
