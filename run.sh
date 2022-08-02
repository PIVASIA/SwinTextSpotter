#swintext
# Download model and save to checkpoints/ 
#setup
cd SwinTextSpotter
conda create -n SWINTS python=3.8 -y
conda activate SWINTS
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python scipy shapely rapidfuzz timm Polygon3
python setup.py build develop

#run predict
python demo/merge.py --config-file projects/SWINTS/configs/SWINTS-swin-finetune-vintext.yaml --input /image --inputfile /input/sign --output ../output/output_merge/ --opts MODEL.WEIGHTS checkpoints/model_final.pth