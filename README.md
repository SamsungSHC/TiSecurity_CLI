# TiSecurity_CLI
Ti Security CLI开源项目。
# 1.什么是Ti Security，这又是什么?
Ti Security是由卡饭论坛 Hashcake (其他网站网名可能为"SamsungSHC"/"Samsung_SHC")个人制作的反病毒安全软件。
完整的Ti Security包括:
- 四个反病毒机器学习引擎
- 基于S6SSM的SSDT解析技术
- Ring 0级较为完整的主动防御系统
- 基于Webview2的UI
完整的Ti Security为闭源软件。此项目为Ti Security的扫描器部分(但不保证具体实现一定相同)项目。它包含:
- 一套可以作为便携式扫描器使用的检测、推理系统
- 四个反病毒机器学习引擎(可能比主线更新晚一些)
你可以将模型、脚本集成到你的项目，但需要遵守许可证限制。

# 2.这四个引擎都分别是什么?
**卷积核(conv)引擎**:
- 基于S6SSM
- 传统的统计数据类特征引擎
- 具有较为可靠的辨别能力
**智芯(zx)引擎**:
- 基于S6SSM
- 借助文件头部判别
- 极速
**FML 引擎**:
- 基于Transformer
- 使用自研的TriGate Sparse Attention(TSA)注意力
- 不局限于头部
**TEX 引擎**:
- 基于TabNet
- 使用VOCAB词表
- 具有较"尖锐"的泛化

四个引擎共享样本集。2026/9/12版本时已达510k样本以上。

# 3.其他
Ti Security 已接入 BrewTotal 平台。你可以通过此站点调用引擎进行在线扫描。
URL:**https://brewtotal.pages.dev(分流)**
URL:**https://brew.virusmark.com(主站)**

你可以发送电子邮件至** 1828529634@qq.com(qq同号)**来联系作者。