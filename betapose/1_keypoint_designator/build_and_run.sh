
mkdir -p build

cd build
rm -r *
cmake ../
make
./pcl-sift ../assets/models/psp.ply ../assets/sifts/sift_psp.ply
# sz ../../assets/sifts/sift_obj01.ply
