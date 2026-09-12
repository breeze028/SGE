# SGE

## 中文说明

这个仓库是在原始论文代码仓库基础上的二次开发版本。原始代码主要支持 triangle soup 形式的几何表示；在此基础上，本仓库扩展了对 mesh 和 texture 的支持，并尝试优化几何重建质量。

需要说明的是，目前的重建效果还不理想，也没有达到原始论文中展示的效果。主要原因是原始代码并没有完整展示论文中高质量结果的具体实现路径，因此这里的实现更多是基于论文、原始仓库和实验理解进行的复现与扩展。

目前这个仓库仅作为学术研究和个人学习用途。代码质量还比较差，工程结构、可维护性和稳定性都需要继续改进。后续计划使用 AI 辅助对代码进行重构，继续提升重建质量，尝试 scale up 到更大规模的场景，并探索更多可微渲染相关课题。

## English

This repository is a secondary development version based on the original paper repository. The original code mainly supports geometry represented as triangle soup. On top of that, this repository adds support for mesh and texture, and also attempts to improve the quality of geometry reconstruction.

Please note that the current reconstruction quality is still limited and does not reach the results shown in the original paper. The main reason is that the original code does not fully demonstrate how the high-quality results in the paper were achieved, so this implementation is more of a reproduction and extension based on the paper, the original repository, and experimental understanding.

At the moment, this repository is intended only for academic research and personal learning. The code quality is still poor, and the engineering structure, maintainability, and stability all need further improvement. In the future, I plan to use AI assistance to refactor the code, improve reconstruction quality, scale up to larger scenes, and explore more topics related to differentiable rendering.
