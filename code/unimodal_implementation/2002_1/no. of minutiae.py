import os
import vault_create2
import quantize2



def vault(file_path, points_list, factor,p):
    """Placeholder function to process minutiae points."""
    print(f"Processing {file_path} with {len(points_list)} minutiae points.")
    # Implement your actual vault logic here
    file_name = file_path.split("\\")[-1].replace(".jpg","")
    #file_path= file_path.split(".")[0].replace("\\","")
    vault_create2.main(points_list, file_name,factor,p)

def process_minutiae(root_dir):


    mini,maxi= 400,-1
    s1=""
    s2=""
    """Traverses directories and processes the first file per user in each subdirectory."""
    f= [8,35,2]
    p= [5, 12,1 ]
    for dirpath, _, filenames in os.walk(root_dir):
        user_files = {}
        
        for file in sorted(filenames):  # Sorting ensures consistent selection
            if file.endswith('.txt'):
                user_id = file.split('_')[0]  # Extract user ID from filename
                if user_id not in user_files:
                    user_files[user_id] = os.path.join(dirpath, file)



        for user_id, file_path in user_files.items():
            minu = [] 
            try:
                with open(file_path, 'r') as file2:
                    for line in file2:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) == 3:
                                x, y, t = map(int, parts)
                                if x<0 or y<0 or t<0 :
                                    continue
                                minu.append([x,y,t])
                        except Exception as e:
                            print(f"[ERROR] Failed to process line in file: {file_path}")
                            print(f"  Line: {line.strip()}")
                            print(f"  Error: {e}")
            except Exception as e:
                print(f"[ERROR] Failed to read file: {file_path}")
                print(f"  Error: {e}")
            for factor in range(f[0],f[1],f[2]):
                #factor = i*3
                minutiae_points = set(quantize2.read_minutiae(minu,factor))
                l= len(minutiae_points)
                if l<mini:
                    mini=l
                    s1= user_id
                if l> maxi:
                    maxi=l
                    s2= user_id
        print(mini,maxi)
        print(s1,s2)

                #vault(file_path, minutiae_points,str(factor),p)

if __name__ == "__main__":
    minutiae_folder = r"minutiae"  # Root directory of minutiae files
    process_minutiae(minutiae_folder)
