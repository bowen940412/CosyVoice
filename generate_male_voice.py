import sys
import os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

def generate_male_voice(text, output_file):

    # 初始化CosyVoice2模型
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
    
    # 加载示例音频作为提示音频
    prompt_speech_16k = load_wav('./asset/cukou01.wav', 16000)
    
    # 使用instruct2模式生成男性声音
    for i, audio_data in enumerate(cosyvoice.inference_instruct2(
        text,
        '用成熟男性的声音说这句话，语气自然平和',  # 指示模型使用男性声音
        prompt_speech_16k,
        stream=False
    )):
        # 保存生成的音频
        torchaudio.save(output_file, audio_data['tts_speech'], cosyvoice.sample_rate)
        print(f"已生成音频文件：{output_file}")

def process_txt_file(txt_file_path):
    # 确保output目录存在
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取txt文件
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 获取文件名（不包含扩展名）作为输出文件名
        base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}.wav")
        
        # 生成语音
        generate_male_voice(text, output_file)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法：python generate_male_voice.py <txt文件路径>")
        sys.exit(1)
    
    txt_file_path = sys.argv[1]
    process_txt_file(txt_file_path) 