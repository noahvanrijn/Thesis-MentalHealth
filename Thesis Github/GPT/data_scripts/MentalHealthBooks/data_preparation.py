
new_file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/MentalHealthBooks_5_percent.txt'

def open_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()
    
text_file = open_file(new_file_path)
print('hi')
print(text_file[:20])




# Function to extract and save the first 5% of lines from a file
def extract_and_save_5_percent(original_file_path, new_file_path):
    with open(original_file_path, 'r', encoding='utf-8') as original_file:
        lines = original_file.readlines()  # Read all lines in the file
    
    # Calculate the number of lines to extract (5% of total lines)
    num_lines_to_extract = len(lines) // 1500  # Integer division by 20 is equivalent to calculating 5%
    
    # Extract the first 5% of the lines
    # You could also choose a different strategy, like evenly distributing the extraction throughout the file
    lines_to_write = lines[:num_lines_to_extract]
    
    # Write the extracted lines to a new file
    with open(new_file_path, 'w', encoding='utf-8') as new_file:
        new_file.writelines(lines_to_write)

# Usage
original_file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/original_datasets/MentalHealthBooks.txt'
new_file_path = '/Users/noahvanrijn/python-repos/master/Thesis/datasets/prepared_datasets/MentalHealthBooks_5_percent.txt'
#extract_and_save_5_percent(original_file_path, new_file_path)
