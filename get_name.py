import os  
  
def write_file_names_to_file(folder_path, output_file):  
    aaa= 'http://jssz-inner-boss.bilibili.co/cv_data_storage/age_estimation/test_jsy_0326_underage/'
    with open(output_file, 'w', encoding='utf-8') as f:  
        for root, dirs, files in os.walk(folder_path):  
            for file in files:  
                bbb = f'{aaa}{file}'
                f.write(bbb + '\n')  
  
# 使用示例  
folder_to_traverse = 'results/test_jsy_0326/test_jsy_0326_underage'  # 替换为你的文件夹路径  
output_txt_file = 'test2.txt'  # 输出文件的名称  
write_file_names_to_file(folder_to_traverse, output_txt_file)