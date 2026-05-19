import gc
import re
import sentencepiece as spm
from fastcdc import fastcdc
from tqdm import tqdm
import os
import time
import math
import hashlib
from collections import defaultdict
import ast
from FeatureSplitTree import build_triple_tree,find_most_common_neighbors_triple
import jpype
import jpype.imports
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import multiprocessing
import sys
import itertools
import math
import csv
from collections import Counter


def wrap_main_with_for_loop():
    # JAR包的路径列表
    jar_paths_list = ['./jar/vcdiff-core-0.1.1.jar', './jar/slf4j-api-1.7.25.jar']
    # 使用os.pathsep来获取当前操作系统的路径分隔符
    classpath = os.pathsep.join(jar_paths_list)
    # 启动JVM，并添加JAR包到classpath
    # jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + classpath)
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        '-ea',
        '-Djava.class.path=' + classpath,
        '-Xms100G',
        '-Xmx160G',
        '-XX:MaxMetaspaceSize=40G',
        '-XX:MetaspaceSize=10G'
    )

    # 计算所有可能的组合
    combinations = list(itertools.product(fsize_ranges, fvocab_size, fmax_sentence_length))
    # # 创建csv目录（如果不存在）
    # csv_dir = os.path.dirname(csv_file_path)
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)
    #
    # # 创建CSV文件并写入列名
    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(['size_range', 'vocab_size', 'max_sentence_length', 'Compression_Rate', 'bits/base_Rate'])
    # 嵌套for循环
    for index, (size_range, vocab, sentence_length) in enumerate(combinations):
        model_name=f'spm(5VU335GB,{vocab},{sentence_length},bpe,{size_range})'
        print(model_name)
        # 重定向标准输出和标准错误输出到文件
        sys.stdout = open(f'CMDResult/size_{size_range}_vocab_{vocab}_length_{sentence_length}_iteration_{index}.txt','w')
        sys.stderr = sys.stdout
        print(model_name)
        # 打印参数到文件中（可选）
        print(f'Model Name: {model_name} Size Range: {size_range}, Vocab Size: {vocab}, Max Sentence Length: {sentence_length}, Iteration: {index}')

        # 执行main函数
        main(size_range=size_range, vocab_size=vocab, max_sentence_length=sentence_length,model_name=model_name)

        # 关闭重定向的文件
        sys.stdout.close()

        # 将标准输出和标准错误输出恢复到控制台
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # 清空控制台（仅在PyCharm中有效）
        os.system('cls' if os.name == 'nt' else 'clear')

    jpype.shutdownJVM()


#读取文件内容放入变量并返回
def read_file(file_path):
    print('read file start')
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    print('read file end')
    return file_content


# 写入新文件函数
def write_file(content, output_file_path):
    print('write file start')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print('write file end')


# 处理文件内容，只保留ATCG并转换为大写
def filtered_DNA(file_content,output_file_path):
    print('filteredDNA start')
    # 只保留ATCG字符，并转换为大写
    filtered_content = re.sub(r'[^ATCGatcg]', '', file_content).upper()
    write_file(filtered_content,output_file_path)
    print('filteredDNA end')
    return filtered_content

# 将处理后的内容转换为二进制表示(从文件读)
def get_binary_from_file(file_path):
    print('read binary file start')
    dna_content=read_file(file_path)

    print('read binary file end')
    return dna_content


def diff_encoding(file_content, output_file_path):
    # 统计词频
    counter = Counter(file_content)
    # 按词频排序
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # 创建编码映射
    encoding = {nucleotide: index for index, (nucleotide, _) in enumerate(sorted_counter)}
    print(encoding)

    # 编码序列
    encoded_content = ''
    for i in range(len(file_content)):
        if i == 0:
            encoded_content += str(format(encoding[file_content[i]],'02b'))
        else:
            distance = encoding[file_content[i]] - encoding[file_content[i - 1]]
            if distance < 0:
                distance += 4
            encoded_content += format(distance, '02b')

    write_file(encoded_content, output_file_path)
    return encoded_content


# # 将处理后的内容转换为二进制表示
# def convert_to_binary(file_content,output_file_path):
#     print('convert to binary start')
#     conversion_dict = {
#         'A': '00',
#         'G': '01',
#         'C': '10',
#         'T': '11'
#     }
#     binary_content = ''.join(conversion_dict[char] for char in tqdm(file_content,desc='processing(convert_to_binary)'))
#     write_file(binary_content,output_file_path)
#     print('convert to binary end')
#     return binary_content


#FastCDC
def fastcdc_method(file_path,size_ranges):
    print('fastcdc start')
    print('max_val:', size_ranges[1], 'min_val:', size_ranges[0])
    fastcdc_results = list(fastcdc(file_path, max_size=size_ranges[1], min_size=size_ranges[0]))
    fastcdc_results_pieces = []
    with open(file_path, "r") as file:
        for fastcdc_result in fastcdc_results:
            offset = fastcdc_result.offset
            size = fastcdc_result.length
            file.seek(offset)
            content = file.read(size)
            fastcdc_results_pieces.append(content)
    print('fastcdc length:', len(fastcdc_results_pieces))
    print('fastcdc end')
    write_file(str(fastcdc_results_pieces), './result/fastcdc_results_pieces.txt')
    return fastcdc_results_pieces
    # for min_val, max_val in size_ranges:
    #     print('max_val:',max_val,'min_val:',min_val)
    #     fastcdc_results = list(fastcdc(file_path, max_size=max_val, min_size=min_val))
    #     fastcdc_results_pieces = []
    #     with open(file_path, "r") as file:
    #         for fastcdc_result in fastcdc_results:
    #             offset = fastcdc_result.offset
    #             size = fastcdc_result.length
    #             file.seek(offset)
    #             content = file.read(size)
    #             fastcdc_results_pieces.append(content)
    #     print('fastcdc length:',len(fastcdc_results_pieces))
    #     print('fastcdc end')
    #     write_file(str(fastcdc_results_pieces),'./result/fastcdc_results_pieces.txt')
    #     return fastcdc_results_pieces


#分为x个等长子块，除不尽就填在最后一个块后面
def sub_chunk(fastcdc_results_pieces,dimension_num):
    print('subchunk start')
    subchunk_list=[]
    for i,content in enumerate(tqdm(fastcdc_results_pieces,desc="Processing pieces")):
        length=len(content)
        subChunkSize=length // dimension_num
        remnant=length % dimension_num
        contents=[]
        for j in range(dimension_num):
            start = j * subChunkSize
            end = (j + 1) * subChunkSize
            if j == dimension_num - 1:
                end += remnant
            contents.append(content[start:end])
        subchunk_list.append(contents)
    print('subchunk end')
    print(len(subchunk_list))
    return subchunk_list


#训练BPE模型
def train_bpe(convert_to_binary_content,model_prefix,model_type,vocab_size, max_sentence_length,num_threads,output_file_path):
    size = len(convert_to_binary_content)
    start = 0
    end = max_sentence_length
    print(max_sentence_length)
    processed_content = ""
    with open(output_file_path, 'w') as outfile:
        while end < size:
            content = convert_to_binary_content[start:end]
            outfile.write(content)
            outfile.write('\n')
            start += max_sentence_length
            end = min(start + max_sentence_length, size)

            # Print progress
            progress = (start / size) * 100
            # print(f"Progress: {progress:.2f}%")

        content = convert_to_binary_content[start:end]
        outfile.write(content)
        # Print progress
        # progress = (start / size) * 100
        # print(f"Progress: {progress:.2f}%")

    print(f"Processing complete. Output written to {output_file_path}")

    print('BPE train start')
    spm.SentencePieceTrainer.train(input=output_file_path, model_prefix=model_prefix, model_type=model_type, vocab_size=vocab_size, max_sentence_length=max_sentence_length,num_threads=num_threads,train_extremely_large_corpus='true')
    # spm.SentencePieceTrainer.train(input=output_file_path, model_prefix=model_prefix, model_type=model_type, vocab_size=vocab_size, max_sentence_length=max_sentence_length)
    # spm.SentencePieceTrainer.train(input=output_file_path, model_prefix=model_prefix, vocab_size=vocab_size, max_sentence_length=max_sentence_length,train_extremely_large_corpus='true',num_threads=35)
    print('BPE train end')


# #bpe特征替换（单线程）
# def bpe_encode_as_piece(subchunk_list, model_path, output_file_path):
#     print('BPE convert start')
#     # 载入模型
#     spm_model = spm.SentencePieceProcessor()
#     spm_model.Load(model_path)
#     subChunk_tokens_list=[]
#     for subchunk in tqdm(subchunk_list,desc="Processing(bep_encode_as_piece"):
#         subchunk_tokens=[]
#         for item in subchunk:
#            subchunk_tokens.append(spm_model.EncodeAsPieces(item))
#         subChunk_tokens_list.append(subchunk_tokens)
#     write_file(str(subChunk_tokens_list), output_file_path)
#     print('BPE convert end')
#     return subChunk_tokens_list


def process_subchunk(index, spm_model, subchunk):
    return index, spm_model.EncodeAsPieces(subchunk)

# #bpe特征替换（多进程）
def bpe_encode_as_piece(subchunk_list, model_prefix, output_file_path):
    print('BPE convert start')
    spm_model_path = model_prefix
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(spm_model_path)
    subChunk_tokens_list = [None] * len(subchunk_list)
    num_workers = 30  # 尝试不同的进程数量
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_subchunk, index, spm_model, subchunk) for index, subchunk in
                   enumerate(subchunk_list)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting results"):
            index, result = future.result()
            subChunk_tokens_list[index] = result

    # 写文件
    write_file(str(subChunk_tokens_list), output_file_path)

    print('BPE convert end')
    return subChunk_tokens_list


# # bpe特征替换(线程)
# def bpe_encode_as_piece(subchunk_list, model_prefix, output_file_path, num_threads=4):
#     print('BPE convert start')
#
#     # 载入模型
#     spm_model = spm.SentencePieceProcessor()
#     spm_model.Load(model_prefix)
#
#     subChunk_tokens_list = [None] * len(subchunk_list)  # 预先分配空间以确保顺序
#
#     def encode_subchunk(index, subchunk):
#         subchunk_tokens = [spm_model.EncodeAsPieces(item) for item in subchunk]
#         return index, subchunk_tokens
#
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         futures = {executor.submit(encode_subchunk, i, subchunk): i for i, subchunk in
#                    enumerate(subchunk_list[:num_threads])}
#
#         with tqdm(total=len(subchunk_list), desc="Processing (bpe_encode_as_piece)") as pbar:
#             for i in range(len(subchunk_list)):
#                 done_future = next(as_completed(futures))
#                 index, subchunk_tokens = done_future.result()
#                 subChunk_tokens_list[index] = subchunk_tokens
#                 pbar.update(1)
#
#                 # Remove the completed future
#                 futures.pop(done_future)
#
#                 # Submit the next subchunk if available
#                 next_index = num_threads + i
#                 if next_index < len(subchunk_list):
#                     futures[executor.submit(encode_subchunk, next_index, subchunk_list[next_index])] = next_index
#
#     write_file(str(subChunk_tokens_list), output_file_path)
#     print('BPE convert end')
#
#     return subChunk_tokens_list

# 读取vocab文件并获取权重
def read_vocab(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return {index + 1: word.strip() for index, word in enumerate(lines) if word.strip()}


# 计算字符串的哈希值并根据权重调整
def calculate_weighted_hash(word, weights_dict):
    # print(word)
    # 获取权重
    weight=find_key(weights_dict,word)
    # print(weight)
    # 计算16位十六进制哈希值
    hash_object = hashlib.md5(word.encode())
    # print(hash_object.hexdigest())
    # print('len:',len(hash_object.hexdigest()))
    hex_digest = hash_object.hexdigest()
    # print('hex_digest:',hex_digest)
    # print('len:',len(hex_digest))
    binary_digest = bin(int(hex_digest, 16))[2:].zfill(128)
    # print('binary_digest:',binary_digest)
    # print('len:',len(binary_digest))
    # 根据权重调整哈希值
    sub_token_weighted_hash=[(weight+1 if bit == '1' else -weight) for bit in binary_digest]
    # print(sub_token_weighted_hash)

    return sub_token_weighted_hash


# 处理文件内容，提取对应关系并形成字典
def extract_mapping(content):
    # 正则表达式匹配键值对
    pattern = re.compile(r'(\d+):\s*\'([^\']+?)\\t')
    matches = pattern.findall(str(content))
    # print(matches)

    mapping_dict = {}
    for key, value in matches:
        mapping_dict[value] = int(key)

    return mapping_dict


# 根据字符串查找对应的数字
def find_key(mapping_dict, target_string):
    return mapping_dict.get(target_string)


# 处理small列表
def process_small(small, weights):
    # print('small(len):',len(small))
    weighted_hashes = [calculate_weighted_hash(word, weights) for word in small if word != '▁']
    # write_file(str(weighted_hashes),'./result/weighted_hashes.txt')
    # print(weighted_hashes)
    # print('lem:',len(weighted_hashes))
    weighted_hashes_sum=sum_weighted_hashes(weighted_hashes)
    # print(weighted_hashes_sum)
    # print('len(weighted_hashes_sum):',len(weighted_hashes_sum))
    processed_result = [1 if value >= 0 else 0 for value in weighted_hashes_sum]
    # print(processed_result)
    # print('len(processed_result):',len(processed_result))

    return processed_result


def sum_weighted_hashes(weighted_hashes):
    # 创建一个空列表，用于存储逐位相加后的结果
    result = []

    # 对于每个子列表，逐位相加并添加到结果列表中
    for bits in zip(*weighted_hashes):
        summed_bits = sum(bits)
        result.append(summed_bits)
    return result


def process_medium(medium, weights_dict):
    medium_after = []
    for small in medium:
        small_hash = process_small(small, weights_dict)
        medium_after.append(small_hash)
    return medium_after

#转hash（多进程）
def convert_to_hash(subchunk_tokens_list,model_name):
    print('convert_to_hash start')
    weights = read_vocab(model_name+'_before.vocab')
    print('weights get')
    weights_dict = extract_mapping(weights)
    print('weights_dict get')

    subchunk_hash_list_after = [None] * len(subchunk_tokens_list)

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        futures = {executor.submit(process_medium, subchunk_tokens_list[i], weights_dict): i for i in
                   range(len(subchunk_tokens_list))}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(subchunk_tokens_list),
                           desc="Processing subchunks"):
            index = futures[future]
            try:
                subchunk_hash_list_after[index] = future.result()
            except Exception as e:
                print(f"Task {index} failed with exception: {e}")
    executor.shutdown(wait=True)
    print("Processing subchunks END!")
    write_file(str(subchunk_hash_list_after), './result/subchunk_hash_list_after.txt')
    print('convert_to_hash end')
    return subchunk_hash_list_after


# def convert_to_hash(subchunk_tokens_list,model_name):
#     print('convert_to_hash start')
#     weights = read_vocab(model_name+'_before.vocab')
#     weights_dict = extract_mapping(weights)
#
#     subchunk_hash_list_after = []
#     for medium in tqdm(subchunk_tokens_list, desc="Processing subchunks"):
#         # 创建新的medium_after列表
#         medium_after = []
#         for small in medium:
#             # print('small:',small)
#             # 计算small的累加哈希值
#             small_hash = process_small(small, weights_dict)
#             # 将哈希值添加到medium_after
#             medium_after.append(small_hash)
#         # 将medium_after添加到big_after
#         subchunk_hash_list_after.append(medium_after)
#     write_file(str(subchunk_hash_list_after),'./result/subchunk_hash_list_after.txt')
#     print('convert_to_hash end')
#     return subchunk_hash_list_after


def calculate_md5(binary_string):
    md5 = hashlib.md5()
    md5.update(binary_string.encode('utf-8'))
    return md5.hexdigest()


def hex_string_to_hex_value(hex_string):
    # 将16进制字符串转换为16进制数值
    hex_value = int(hex_string, 16)
    # print(hex_string,'/',hex_value)
    return hex_value


def process_subchunk_super_feature(subchunk_list, dimension_num, group_num):
    super_feature_list = []

    # Step 1: Convert each dimension upperdata into binary strings
    binary_data = []
    for subchunk in subchunk_list:
        # 将子列表转换为字符串并添加到结果列表中
        concatenated = ''.join(map(str, subchunk))
        # print('concatenated:',concatenated)
        binary_data.append(concatenated)
    # print('binary_data:', binary_data)
    # Step 2: Divide the binary_data into groups
    start=0
    end=start
    binary_data_group_list=[]
    group_size=int(dimension_num/group_num)
    for i in range(group_num):
        temp=[]
        for j in range(group_size):
            temp = binary_data[i*group_size:(i+1)*group_size]
            temp = sorted(temp, reverse=True)
            # print('temp(after):',temp)
        binary_data_group_list.append(temp)
    # write_file(str(binary_data_group_list),'./result/binary_data_group_list.txt')
    #superfeature

    # Step 3: Process each group
    super_fearure_group = []
    for i in range(group_size):
        concatenated_string = ""
        for group in binary_data_group_list:
            # Concatenate elements at the same index
            concatenated_string += group[i]
            # print('concatenated_string:',concatenated_string)
            # write_file(concatenated_string,'./result/concatenated_string.txt')
        # Calculate MD5 hash of the concatenated string
        md5_hash = hashlib.md5(concatenated_string.encode()).hexdigest()
        super_fearure_group.append(hex_string_to_hex_value(md5_hash))
        # super_fearure_group.append(md5_hash)
    return super_fearure_group
    # write_file(str(super_fearure_group),'./result/super_fearure_group.txt')
    # # print('super_fearure_group:',super_fearure_group)
    #
    #     # print('binary_data_group_list:',binary_data_group_list)
    # write_file(str(binary_data_group_list),'./result/binary_data_group_list.txt')


def process_all_subchunks_super_feature(subchunk_hash_list_after, dimension_num, group_num):
    all_super_features = []
    for subchunk in subchunk_hash_list_after:
        # write_file(str(subchunk),'./result/subchunks.txt')
        super_features = process_subchunk_super_feature(subchunk, dimension_num, group_num)
        all_super_features.append(super_features)
    write_file(str(all_super_features),'./result/all_subchunks_super_feature_list.txt')
    print(len(all_super_features))
    return all_super_features

def get_width_dictionary(all_subchunks_super_feature_list):
    # tree = build_tree(all_subchunks_super_feature_list)
    tree1,tree2,tree3 = build_triple_tree(all_subchunks_super_feature_list)
    width_dictionary_chunks = find_most_common_neighbors_triple(tree1,tree2,tree3,all_subchunks_super_feature_list,16)
    # width_dictionary_chunks = find_most_common_neighbors(tree,all_subchunks_super_feature_list,16)
    width_dictionary_chunks_ids=[]
    for chunk in width_dictionary_chunks:
        width_dictionary_chunks_ids.append(chunk[0])
    return width_dictionary_chunks_ids


def get_deep_dictionary(all_subchunks_super_feature_list,subchunk_num):
    deep_dictionary_chunk_ids=[]
    length=len(all_subchunks_super_feature_list)
    remnant=length % subchunk_num
    size_num=length // subchunk_num
    start=0
    end=size_num
    for i in range(subchunk_num):
        temp = all_subchunks_super_feature_list[start:end]
        # tree=build_tree(temp)
        tree1,tree2,tree3 = build_triple_tree(temp)
        a = find_most_common_neighbors_triple(tree1,tree2,tree3,temp,1)[0][0]+start
        # a = find_most_common_neighbors(tree, temp, 1)[0][0]+start
        # print(a)
        deep_dictionary_chunk_ids.append(a)
        start=start+size_num
        end=end+size_num
        if i == subchunk_num-2:
            end=end+remnant
    return deep_dictionary_chunk_ids


def merge_and_sort_dictionary_chunk_ids(list1, list2):
    # 合并两个列表
    merged_list = list1 + list2
    # 去除重复元素并排序
    unique_sorted_list = sorted(set(merged_list))
    write_file(str(unique_sorted_list),'./result/deep_width_dictionary_chunk_ids.txt')
    return unique_sorted_list


def get_vcdiff_dictionary(fastcdc_results_pieces, indices):
    # 提取对应索引的元素并拼接在一起
    concatenated_elements = ''.join([fastcdc_results_pieces[i] for i in indices])
    write_file(concatenated_elements,'./result/vcdiff_dictionary.txt')
    print('vcdiff dic size:',os.path.getsize('./result/vcdiff_dictionary.txt'))
    return concatenated_elements


def vcdiff_compression(vcdiff_dictionary,convert_to_binary_content):
    # # JAR包的路径列表
    # jar_paths_list = ['./jar/vcdiff-core-0.1.1.jar', './jar/slf4j-api-1.7.25.jar']
    # # 使用os.pathsep来获取当前操作系统的路径分隔符
    # classpath = os.pathsep.join(jar_paths_list)
    # # 启动JVM，并添加JAR包到classpath
    # # jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=' + classpath)
    # jpype.startJVM(
    #     jpype.getDefaultJVMPath(),
    #     '-ea',
    #     '-Djava.class.path=' + classpath,
    #     '-Xms100G',
    #     '-Xmx160G',
    #     '-XX:MaxMetaspaceSize=40G',
    #     '-XX:MetaspaceSize=10G'
    # )
    Runtime = jpype.java.lang.Runtime.getRuntime()

    # 输出JVM的内存信息
    print("Total Memory: ", Runtime.totalMemory() / (1024 * 1024), "MB")
    print("Free Memory: ", Runtime.freeMemory() / (1024 * 1024), "MB")
    print("Max Memory: ", Runtime.maxMemory() / (1024 * 1024), "MB")
    # 导入VCDiffEncoderBuilder类
    VCDiffEncoderBuilder = jpype.JClass('com.davidehrmann.vcdiff.VCDiffEncoderBuilder')

    #创建VCDIFF编码器
    print('create vcdiff encoder start')
    encoder_builder = VCDiffEncoderBuilder.builder()
    encoder = encoder_builder.withDictionary(vcdiff_dictionary).buildSimple()
    print('create vcdiff encoder end')
    # block = 'niu'
    # blocks_content_bytes = [block.encode('utf-8')]
    # dictionary = b''.join(blocks_content_bytes)
    # encoder = encoder_builder.withDictionary(dictionary).buildSimple()

    # 使用ByteArrayOutputStream来捕获压缩数据
    # sys.set_int_max_str_digits(0)
    compressed_output = jpype.JClass('java.io.ByteArrayOutputStream')()
    print('convert to JArray start')
    # 对pieces列表中的每个块使用这个词典进行压缩
    # 将Python bytes转换为Java byte数组
    java_data = jpype.JArray(jpype.JByte)(convert_to_binary_content)
    print('convert to JArray end')
    print('vcdiff encode start')
    # 编码数据
    encoder.encode(java_data, compressed_output)
    print('vcdiff encode end')
    print('convert to python bytes start')
    # 将压缩数据转换回Python bytes
    compressed_data = bytes(compressed_output.toByteArray())
    print('convert to python bytes end')

    write_file(str(compressed_data),'./result/vcdiff_compression_result.txt')

    print('after-vcdiff-compression_file-size:',os.path.getsize('./result/vcdiff_compression_result.txt'))
    # 关闭JVM
    # jpype.shutdownJVM()

    return compressed_data


def list_to_bin(model_name,data_list):
    byte_array = bytearray()

    # Convert each element to a single byte and append to byte_array
    for num in data_list:
        byte_array.append(num % 256)

    # Write the byte_array to a file
    with open(f'./result/output{model_name}.bin', 'wb') as file:
        file.write(byte_array)

    # Print the size of the stored file
    print('bpe_result存储的文件大小为：' + str(os.path.getsize(f'./result/output{model_name}.bin')) + "字节")

def main(size_range,vocab_size,max_sentence_length,model_name):
    start_time=time.time()
    original_file_content = read_file(original_file_path)
    read_original_file_spend_time=time.time()
    print('read original file spend:',read_original_file_spend_time-start_time)
    print('total spend:',read_original_file_spend_time-start_time)
    print('原始文件大小',os.path.getsize(original_file_path))


    filtered_DNA_content=filtered_DNA(original_file_content,filtered_DNA_file_path)
    filtered_DNA_spend_time=time.time()
    print('filtered_DNA_spend_time:',filtered_DNA_spend_time-read_original_file_spend_time)
    print('total spend:',filtered_DNA_spend_time-start_time)
    print('过滤DNA后文件大小',os.path.getsize(filtered_DNA_file_path))


    convert_to_binary_content=diff_encoding(filtered_DNA_content,convert_to_binary_file_path)
    diff_encoding_spend_time=time.time()
    print('diff_encoding_spend_time:',diff_encoding_spend_time-filtered_DNA_spend_time)
    print('total spend:',diff_encoding_spend_time-start_time)
    # convert_to_binary_content=get_binary_from_file(convert_to_binary_file_path)
    print('差分后后文件大小',os.path.getsize(convert_to_binary_file_path))


    # #FastCDC分块（不转01串）
    # fastcdc_results_pieces=fastcdc_method(filtered_DNA_file_path,size_ranges)

    #FastCDC分块（转01串）
    fastcdc_results_pieces=fastcdc_method(convert_to_binary_file_path,size_range)
    fast_cdc_spend_time=time.time()
    print('fast_cdc_spend_time:',fast_cdc_spend_time-diff_encoding_spend_time)
    print('total spend:',fast_cdc_spend_time-start_time)


    #维度分块
    subchunk_list=sub_chunk(fastcdc_results_pieces,12)
    sub_chunk_spend_time=time.time()
    print('sub_chunk_spend_time:',sub_chunk_spend_time-fast_cdc_spend_time)
    print('total spend:',sub_chunk_spend_time-start_time)

    # 训练SentencePiece模型
    train_bpe(convert_to_binary_content,model_name+'_before','bpe',vocab_size,max_sentence_length,32,'./result/convert_to_dinary_add_linebreak(before).txt')
    train_bpe_spend_time=time.time()
    print('train_bpe_spend_time:',train_bpe_spend_time-sub_chunk_spend_time)
    print('total spend:',train_bpe_spend_time-start_time)

    #BPE词典编码
    subchunk_tokens_list=bpe_encode_as_piece(subchunk_list, model_name+'_before.model', token_list_file_path)
    bpe_encode_as_piece_spend_time=time.time()
    print('bpe_encode_as_piece消耗时间:',bpe_encode_as_piece_spend_time-train_bpe_spend_time)
    print('total spend:',bpe_encode_as_piece_spend_time-start_time)

    subchunk_hash_list_after=convert_to_hash(subchunk_tokens_list,model_name)
    convert_to_hash_spend_time=time.time()
    print('convert_to_hash消耗时间:',convert_to_hash_spend_time-bpe_encode_as_piece_spend_time)
    print('total spend:',convert_to_hash_spend_time-start_time)

    all_subchunks_super_feature_list=process_all_subchunks_super_feature(subchunk_hash_list_after,12,4)
    process_all_subchunks_super_feature_spend_time=time.time()
    print('process_all_subchunks_super_feature消耗时间:',process_all_subchunks_super_feature_spend_time-convert_to_hash_spend_time)
    print('total spend:',process_all_subchunks_super_feature_spend_time-start_time)
    # end_time = time.time()
    # print("时间消耗:",end_time-start_time)

    width_dictionary_chunk_ids = get_width_dictionary(all_subchunks_super_feature_list)
    print('width_dictionary_chunk_ids:',width_dictionary_chunk_ids)
    deep_dictionary_chunk_ids = get_deep_dictionary(all_subchunks_super_feature_list,16)
    print('deep_dictionary_chunk_ids:',deep_dictionary_chunk_ids)
    deep_width_dictionary_chunk_ids=merge_and_sort_dictionary_chunk_ids(width_dictionary_chunk_ids,deep_dictionary_chunk_ids)
    get_width_deep_dictionary_spend_time=time.time()
    print('get_width_deep_dictionary_spend_time:',get_width_deep_dictionary_spend_time-process_all_subchunks_super_feature_spend_time)
    print('total spend:',get_width_deep_dictionary_spend_time-start_time)
    print('deep_width_dictionary_chunk_ids:',deep_width_dictionary_chunk_ids)
    #数据正确性没确定
    vcdiff_dictionary=get_vcdiff_dictionary(fastcdc_results_pieces,deep_width_dictionary_chunk_ids)
    # print('vcdiff_dictionary:',vcdiff_dictionary)
    print('convert_to_binary_content type:',type(convert_to_binary_content))
    print('convert_to_binary_content type(after-encode):',type(convert_to_binary_content.encode('utf-8')))
    vcdiff_compression_result=vcdiff_compression(vcdiff_dictionary.encode('utf-8'),convert_to_binary_content.encode('utf-8'))
    vcdiff_spend_time=time.time()
    print('vcdiff_spend_time:',vcdiff_spend_time-get_width_deep_dictionary_spend_time)
    print('total spend:',vcdiff_spend_time-start_time)

    #after_bpe
    train_bpe(str(vcdiff_compression_result),model_name+'_after','bpe',vocab_size,max_sentence_length,32,'./result/convert_to_dinary_add_linebreak(after).txt')
    after_bpe_train_spend_time=time.time()
    print('after_bpe_train_spend_time:',after_bpe_train_spend_time-vcdiff_spend_time)
    print('total spend:',after_bpe_train_spend_time-start_time)
    # 载入模型
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(model_name+'_after.model')
    bpe_compression_after_result=spm_model.Encode(vcdiff_compression_result)
    after_bpe_encode_spend_time=time.time()
    print('after_bpe_encode_spend_time:',after_bpe_encode_spend_time-after_bpe_train_spend_time)
    print('total spend:',after_bpe_encode_spend_time-start_time)
    write_file(str(bpe_compression_after_result), 'result/bpe_compression_after_result.txt')
    print('after bep compression result',len(bpe_compression_after_result))
    length_result = len(bpe_compression_after_result)
    log_vocab = math.log2(vocab_size)
    length_orgin=os.path.getsize(filtered_DNA_file_path)
    compression_rate = 1-((length_result * (log_vocab / 8))/length_orgin)
    bits_base_rate=((length_result * log_vocab)/length_orgin)
    print('Compression Rate:',compression_rate)
    print('bits/base Rate:',bits_base_rate)
    gc.collect()
    # # 打开CSV文件，以追加模式写入新行
    # with open(csv_file_path, 'a', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([size_range, vocab_size, max_sentence_length, compression_rate, bits_base_rate])


    data=read_file('./result/bpe_compression_after_result.txt')
    data_list = eval(data)
    # print(data_list)
    # print(type(data_list))
    list_to_bin(model_name,data_list)
    list_to_bin_spend_time=time.time()
    # print('list_to_bin_spend_time消耗时间:',list_to_bin_spend_time-after_bpe_encode_spend_time)
    # print('total spend:',list_to_bin_spend_time-start_time)



# original_file_path = 'upperdata/Acanthochromis_polyacanthus.ASM210954v1.dna_sm.nonchromosomal.fa'
# csv_file_path='./CSVResult/HUM.csv'
original_file_path = '/home/card/LX Test/cut_data/spm(VU335GB,256,32768,bpe,(32768, 65536))/5Vombatus_ursinus.bare-nosed_wombat_genome_assembly.dna_sm.nonchromosomal.fa'
filtered_DNA_file_path='./result/filtered_DNA.fasta'
convert_to_binary_file_path='./result/convert_to_dinary.txt'
token_list_file_path="./result/token_list.txt"
fsize_ranges = [(32768,65536)]
fvocab_size=[256]
fmax_sentence_length=[32768]
wrap_main_with_for_loop()

