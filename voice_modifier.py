#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice2 语音修改工具

该脚本提供了多种方法来修改语音声音，包括：
- 零样本语音克隆：使用参考音频的声音特征
- 指令控制：通过文字指令改变语音风格（如方言、情感等）
- 细粒度控制：添加特殊标记来改变语音表现
- 自定义提示音频：使用自己的音频作为声音参考

使用前请确保已下载CosyVoice2-0.5B模型到pretrained_models目录
"""

import sys
import os
import time
from pathlib import Path
import argparse

# 添加Matcha-TTS到系统路径
sys.path.append('third_party/Matcha-TTS')

# 导入必要的库
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 创建输出目录
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

def load_model():
    """加载CosyVoice2模型"""
    print("正在加载CosyVoice2模型...")
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', 
                          load_jit=False, 
                          load_trt=False, 
                          fp16=False)
    print(f"模型加载完成，采样率: {cosyvoice.sample_rate}Hz")
    return cosyvoice

def zero_shot_cloning(cosyvoice, text, prompt_audio_path, prompt_text=None):
    """零样本语音克隆 - 使用参考音频的声音特征"""
    print("\n执行零样本语音克隆...")
    print(f"使用参考音频: {prompt_audio_path}")
    
    # 如果未提供提示文本，使用默认值
    if prompt_text is None:
        prompt_text = "希望你以后能够做的比我还好呦。"
    
    # 加载提示音频
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    
    start_time = time.time()
    output_files = []
    
    for i, result in enumerate(cosyvoice.inference_zero_shot(
            text, 
            prompt_text, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"modified_voice_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        output_files.append(output_path)
        print(f"  已保存到 {output_path}")
    
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    return output_files

def instruction_control(cosyvoice, text, instruction, prompt_audio_path):
    """指令控制 - 通过文字指令改变语音风格"""
    print("\n执行指令控制...")
    print(f"指令: {instruction}")
    print(f"使用参考音频: {prompt_audio_path}")
    
    # 加载提示音频
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    
    start_time = time.time()
    output_files = []
    
    for i, result in enumerate(cosyvoice.inference_instruct2(
            text, 
            instruction, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"modified_voice_instruct_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        output_files.append(output_path)
        print(f"  已保存到 {output_path}")
    
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    return output_files

def fine_grained_control(cosyvoice, text_with_control, prompt_audio_path):
    """细粒度控制 - 添加特殊标记来改变语音表现"""
    print("\n执行细粒度控制...")
    print(f"带控制标记的文本: {text_with_control}")
    print(f"使用参考音频: {prompt_audio_path}")
    
    # 加载提示音频
    prompt_speech_16k = load_wav(prompt_audio_path, 16000)
    
    start_time = time.time()
    output_files = []
    
    for i, result in enumerate(cosyvoice.inference_cross_lingual(
            text_with_control, 
            prompt_speech_16k, 
            stream=False,
            text_frontend=False)):
        output_path = output_dir / f"modified_voice_control_{i}.wav"
        torchaudio.save(str(output_path), result['tts_speech'], cosyvoice.sample_rate)
        output_files.append(output_path)
        print(f"  已保存到 {output_path}")
    
    print(f"  耗时: {time.time() - start_time:.2f}秒")
    return output_files

def print_help():
    """打印帮助信息"""
    print("\n语音修改选项:")
    print("1. 零样本语音克隆 - 使用参考音频的声音特征")
    print("2. 指令控制 - 通过文字指令改变语音风格（如方言、情感等）")
    print("3. 细粒度控制 - 添加特殊标记来改变语音表现")
    print("4. 退出")

def main():
    parser = argparse.ArgumentParser(description='CosyVoice2语音修改工具')
    parser.add_argument('--mode', type=int, help='修改模式: 1=零样本克隆, 2=指令控制, 3=细粒度控制')
    parser.add_argument('--text', type=str, help='要合成的文本')
    parser.add_argument('--prompt_audio', type=str, default='./asset/zero_shot_prompt.wav', help='参考音频路径')
    parser.add_argument('--instruction', type=str, help='指令文本（用于指令控制模式）')
    
    args = parser.parse_args()
    
    # 加载模型
    cosyvoice = load_model()
    
    # 如果提供了命令行参数，直接执行相应模式
    if args.mode and args.text:
        if args.mode == 1:
            zero_shot_cloning(cosyvoice, args.text, args.prompt_audio)
        elif args.mode == 2 and args.instruction:
            instruction_control(cosyvoice, args.text, args.instruction, args.prompt_audio)
        elif args.mode == 3:
            fine_grained_control(cosyvoice, args.text, args.prompt_audio)
        return
    
    # 交互式模式
    while True:
        print_help()
        choice = input("\n请选择修改选项 (1-4): ")
        
        if choice == '4':
            print("退出程序")
            break
        
        text = input("请输入要合成的文本: ")
        prompt_audio = input(f"请输入参考音频路径 (默认: ./asset/zero_shot_prompt.wav): ") or "./asset/zero_shot_prompt.wav"
        
        if choice == '1':
            zero_shot_cloning(cosyvoice, text, prompt_audio)
        elif choice == '2':
            instruction = input("请输入指令 (例如: '用四川话说这句话'): ")
            instruction_control(cosyvoice, text, instruction, prompt_audio)
        elif choice == '3':
            print("\n细粒度控制提示:")
            print("- 使用[laughter]标记添加笑声")
            print("- 更多控制标记请参考文档")
            text_with_control = input("请输入带控制标记的文本: ")
            fine_grained_control(cosyvoice, text_with_control, prompt_audio)
        else:
            print("无效选项，请重新选择")
    
    print("\n所有生成的音频文件保存在 {}".format(output_dir.absolute()))

if __name__ == "__main__":
    main()