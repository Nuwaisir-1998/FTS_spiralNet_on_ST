# python train.py --path=./checkpoints_3/ImagineGAN/dlpfc_train/
# python test.py --path=./checkpoints_3/ImagineGAN/dlpfc_test/
# cp ./checkpoints_3/ImagineGAN/dlpfc_train/imagine_g.pth ./checkpoints_3/SliceGAN/dlpfc_train/imagine_g.pth
# cp ./checkpoints_3/ImagineGAN/dlpfc_train/imagine_g.pth ./checkpoints_3/SliceGAN/dlpfc_test/imagine_g.pth

# make sure you copy the slice_g.pth, and other 2 files to to SliceGAN/dlpfc_test/ as well

# python train.py --path=./checkpoints_3/SliceGAN/dlpfc_train/
python test.py --path=./checkpoints_3/SliceGAN/dlpfc_test/