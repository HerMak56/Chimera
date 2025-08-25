import torch

def socialnav_collate(batch):
    B = len(batch)
    out = {}
    # time-major: past_frames[t][B,3,H,W]
    T = len(batch[0]["past_frames"])
    out["past_frames"] = [torch.stack([batch[b]["past_frames"][t] for b in range(B)]) for t in range(T)]
    # остальные ключи
    for k in ["future_frame","future_positions","future_yaw","future_vw",
              "past_positions","past_yaw","past_vw","goal_direction","dt"]:
        out[k] = torch.stack([torch.as_tensor(batch[b][k]) for b in range(B)])
    if "obs_traversability" in batch[0]:
        out["obs_traversability"] = torch.stack([batch[b]["obs_traversability"] for b in range(B)])
    return out