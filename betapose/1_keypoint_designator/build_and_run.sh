
mkdir -p build

cd build
rm -r *
cmake ../
make
./pcl-sift ../assets/models/kuka.ply ../assets/sifts/sift_kuka.ply
# sz ../../assets/sifts/sift_obj01.ply
