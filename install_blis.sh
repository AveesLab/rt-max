cd ~
sudo apt update
sudo apt install build-essential
git clone https://github.com/flame/blis.git
cd blis
./configure auto
make
sudo make install

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib