wget -O public.h5py 'https://www.dropbox.com/s/k5m08ba3qtibk0d/-257-0.70.h5py?dl=1'

wget -O private.h5py 'https://www.dropbox.com/s/55vsz7178r02qs5/-209-0.69.h5py?dl=1'

sed -i -e 's/, "amsgrad": false/                  /g' 'public.h5py'
sed -i -e 's/, "amsgrad": false/                  /g' 'private.h5py'

python predict.py $1 $2 $3
