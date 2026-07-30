# Model use and distribution policy

Last reviewed: 2026-07-30

K2Lab RunPod is noncommercial open-source software licensed under Apache-2.0.
The software license does not license any model or auxiliary weight.

Source releases and container images must not bundle or redistribute Krea, Qwen, LoRA,
detector, pose-adapter, or upscaler weights, include a prepopulated model cache, or
automatically acquire weights during installation or startup. An operator may explicitly
upload an authorized asset or start a download directly from an authorized upstream
provider into the operator's private workspace. Registry hashes identify compatible
files but grant no rights.

Krea 2 weights are governed by the
[Krea 2 Community License Agreement](https://www.krea.ai/krea-2-licensing) and
[Krea Acceptable Use Policy](https://www.krea.ai/krea-2-use-policy), not Apache-2.0.
Requesting a provider download, uploading, configuring, or using Krea weights is the
operator's direct interaction with those terms; K2Lab does not accept them on the
operator's behalf and does not operate a model mirror.

The tested RunPod configuration is a private, single-operator workspace in which the
operator reviews prompts and outputs. It must not be exposed as a public or shared
generation service without content filtering or an equivalent review process
appropriate to the current Krea terms and the use case.

The reviewed converted Qwen FP8 text encoder has no separately documented conversion
and notice chain. It must not be redistributed unless that chain is confirmed and all
required Apache-2.0 notices are preserved.

Repeat model-license and deployment review before bundling or redistributing weights,
adding a project-operated mirror or unattended model acquisition, changing a reviewed
model source/hash, enabling public or multi-user inference, removing operator review,
or using the service commercially.
