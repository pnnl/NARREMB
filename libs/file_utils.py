import os

def create_dir(path):
    '''
    Create path to directories, if it doesn't exist
    '''
    
    if os.path.exists(path):
        print("Directory already exists...")
    else:
        os.makedirs(path)
        print(f"Created...{path}")