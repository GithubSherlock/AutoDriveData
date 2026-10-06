"""图像 / 场景**编辑**能力面:生成模型的后端与条件源。

与其它能力面的关系(层守卫强制,见 `tests/test_layer_guard.py` 的 `LAYER_RULES`):

- **许 torch**(权重是 torch checkpoint;`cldm`/`ldm` 是纯 Python + torch);
- **禁 carla** —— 条件源吃的是盘上**已采好**的数据(真值深度 / 语义 tag / RGB),
  采集是 `sim/` 的事。这一条与 `perception/` 同形:评测层不连仿真器。

外部模型(StyleGAN3 / ControlNet)的两个官方 repo 克隆在 `hdMapGitHub/`,**保持 pristine**;
版本适配全部收在本包的 `cldm_backend`,不在 repo 里改一行。计划与实测见
[docs/edit-image-plan.md](../../docs/edit-image-plan.md)。
"""
