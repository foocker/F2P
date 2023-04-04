# data_dir=/data/img_align_celeba_crop_frontal
# out_dir=../out_celebA
data_dir=/data/data_320/test/Texture 
out_dir=../out_character_femal_helen_texture

# snapshot=../DML_CSR/dml_csr_celebA.pth
snapshot=../DML_CSR/dml_csr_lapa.pth
# snapshot=../DML_CSR/dml_csr_helen.pth
# python test.py --data-dir "$data_dir" --out-dir "$out_dir" --restore-from "$snapshot" --gpu "0" --batch-size 6 --input-size 256,256 --dataset test --num-classes 19
python test.py --data-dir "$data_dir" --out-dir "$out_dir" --restore-from "$snapshot" --gpu "0" --batch-size 6 --input-size 256,256 --dataset test --num-classes 11