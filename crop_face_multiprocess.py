from K2P.faceanalysis.insightface.iface import FaceInformation
from K2P.fileio.path import mkdir_or_exist
import glob, os, cv2


img_dir="" # your own celeba img_dir
Ff = FaceInformation(detect_only=True)
img_list = sorted(glob.glob(img_dir + "/*.jpg"))
img_align_celeba_crop = os.path.join(os.path.dirname(img_dir),"img_align_celeba_crop")
mkdir_or_exist(img_align_celeba_crop)

def hand_func(img_sublist):
    for imgf in img_sublist:
        img = cv2.imread(imgf)
        try:
            _, aimg, _, _, _ = Ff.info(img)  # may change  _, _
            dst_p = os.path.join(img_align_celeba_crop, os.path.basename(imgf))
            Ff.save_img(aimg, dst_p)
        except:
            print("Wrong")
            continue

def main_process(process_num=8):
    from multiprocessing import Process
    one_split_num = len(img_list) // process_num

    img_lists = [img_list[i*one_split_num:(i+1)*one_split_num] for i in range(process_num+1)]
    process = [Process(target=hand_func, args=(img_lists[i],)) for i in range(process_num)]
    [p.start() for p in process]  
    [p.join() for p in process]

if __name__ == "__main__":
    main_process(process_num=16) 