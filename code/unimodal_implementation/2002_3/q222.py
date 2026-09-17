import os

def int_to_binary(num, length=6):
    return format(num, f'0{length}b')

def read_minutiae(points,factor = 20):
    """Reads minutiae points from a text file."""
    minutiae_points = []
    pp = [] 
    #factor = 20
    try:
        #print(points)
        for parts in points:
                    #print(parts)
                    if len(parts) == 3:
                        x, y, t = map(int, parts)
                        if x<0 or y<0 or t<0 :
                            continue
                        x, y, t = map(int_to_binary, [x//factor, y//factor, t//factor])

                        z= int(x+y,2)

                        if z in pp:
                            continue
                        else:
                            pp.append(z)
                        binary_string = x + y + t
                        minutiae_points.append(int(binary_string, 2))
    except Exception as e:
        #print(f"[ERROR] Failed to process line in file: {file_path}")
        #print(f"  Line: {line.strip()}")
        print(f"  Error: {e}")


    return minutiae_points,pp
