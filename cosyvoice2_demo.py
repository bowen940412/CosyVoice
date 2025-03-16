#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice2 调用示例脚本

该脚本展示了如何使用CosyVoice2模型进行语音合成，包括：
- 零样本语音克隆 (Zero-shot Voice Cloning)
- 细粒度控制 (Fine-grained Control)
- 指令控制 (Instruction-based Synthesis)
- 双向流式推理 (Bidirectional Streaming)

使用前请确保已下载CosyVoice2-0.5B模型到pretrained_models目录
"""

import sys
import os
import time
from pathlib import Path

# 添加Matcha-TTS到系统路径
sys.path.append('third_party/Matcha-TTS')

# 导入必要的库
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 创建输出目录
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

def main():
    print("正在加载CosyVoice2模型...")
    # 初始化CosyVoice2模型
    # load_jit=False: 不使用JIT模式
    # load_trt=False: 不使用TensorRT
    # fp16=False: 不使用半精度浮点数
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', 
                          load_jit=False, 
                          load_trt=False, 
                          fp16=False)
    print(f"模型加载完成，采样率: {cosyvoice.sample_rate}Hz")
    
    # 加载示例提示音频
    print("加载提示音频...")
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    
    # 1. 零样本语音克隆 (Zero-shot Voice Cloning)
    print("\n1. 执行零样本语音克隆...")
    text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    start_time = time.time()
    for i, result in enumerate(cosyvoice.inference_zero_shot(
            text, 
            prompt_text, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"zero_shot_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"  已保存到 {output_path}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    # 2. 细粒度控制 (Fine-grained Control)
    print("\n2. 执行细粒度控制...")
    # [laughter] 标记用于控制笑声
    text_with_control = "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。"
    
    start_time = time.time()
    for i, result in enumerate(cosyvoice.inference_cross_lingual(
            text_with_control, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"fine_grained_control_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"  已保存到 {output_path}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    # 3. 指令控制 (Instruction-based Synthesis)
    print("\n3. 执行指令控制...")
    text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    instruction = "用四川话说这句话"
    
    start_time = time.time()
    for i, result in enumerate(cosyvoice.inference_instruct2(
            text, 
            instruction, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"instruct_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"  已保存到 {output_path}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    # 4. 双向流式推理 (Bidirectional Streaming)
    print("\n4. 执行双向流式推理...")
    # 使用生成器作为输入，适用于与文本LLM模型集成
    def text_generator():
        yield "收到好友从远方寄来的生日礼物，"
        yield "那份意外的惊喜与深深的祝福"
        yield "让我心中充满了甜蜜的快乐，"
        yield "笑容如花儿般绽放。"
    
    start_time = time.time()
    for i, result in enumerate(cosyvoice.inference_zero_shot(
            text_generator(), 
            prompt_text, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"bistream_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"  已保存到 {output_path}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    # 5. 流式推理示例 (Streaming Inference)
    print("\n5. 执行流式推理示例...")
    text = "这是一个流式推理的示例，可以实现低延迟的语音合成，适合实时对话场景。"
    
    start_time = time.time()
    for i, result in enumerate(cosyvoice.inference_zero_shot(
            text, 
            prompt_text, 
            prompt_speech_16k, 
            stream=True,  # 启用流式推理
            text_frontend=False)):
        output_path = output_dir / f"streaming_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        print(f"  已保存流式音频片段到 {output_path}")
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    print("\n所有演示完成！生成的音频文件保存在 {}".format(output_dir.absolute()))

if __name__ == "__main__":
    main()