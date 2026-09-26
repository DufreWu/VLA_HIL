from pathlib import Path


def load_policy(model_path, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(model_path))
    policy.to(device)
    policy.eval()
    return policy
