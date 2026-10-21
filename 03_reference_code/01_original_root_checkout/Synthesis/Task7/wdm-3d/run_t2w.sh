# General settings — T2W‑only version
# Usage:   bash run_t2w.sh           (train or sample depending on MODE below)
SEED=42;
CHANNELS=64;
MODE='train';             # 'train' or 'sample'
DATASET='brats';
MODEL='ours_unet_256';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'
TRAIN_MODE='known_3_to_gen_1';
echo "TRAIN_MODE: ${TRAIN_MODE}  (T2W only)"

# settings for sampling/inference
ITERATIONS=2000;
SAMPLING_STEPS=3000;
RUN_DIR="runs/known_3_to_gen_1_27_7_2024_16:16:14/";

IN_CHANNELS=32;
OUT_CHANNELS=32;

if [[ $TRAIN_MODE == 'known_all_time' ]]; then
  IN_CHANNELS=36;
  OUT_CHANNELS=32;
  TUMOUR_LOSS_WEIGHT=1;
fi

if [[ $TRAIN_MODE == 'known_3_to_gen_1' ]]; then
  IN_CHANNELS=36;
  OUT_CHANNELS=8;
  TUMOUR_LOSS_WEIGHT=1;
fi

echo "IN_CHANNELS: ${IN_CHANNELS}"
echo "OUT_CHANNELS: ${OUT_CHANNELS}"

DATA_DIR=../../DataSet/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData;
DATA_SPLIT_JSON=../utils/BraTS2023-Missing_modal_training_data_split.json;

# detailed settings
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  CHANNEL_MULT=1,2,2,4,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=10;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample (T2W only)"
  BATCH_SIZE=1;
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'brats' ]]; then
    echo "MODE: training (T2W only)";
    echo "DATASET: BRATS";
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=${OUT_CHANNELS}
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
--data_split_json=${DATA_SPLIT_JSON}
--num_workers=8
"

TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--train_mode=${TRAIN_MODE}
--save_interval=5000
--tumour_loss_weight=${TUMOUR_LOSS_WEIGHT}
"

SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--output_dir=./results/${RUN_DIR}/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
--train_mode=${TRAIN_MODE}
--mode=${MODE}
"

# ----------------------------------------------------------------
# Call the T2W-only scripts instead of the originals
# ----------------------------------------------------------------
if [[ $MODE == 'train' ]]; then
  python scripts/generation_train_t2w.py $TRAIN $COMMON;
else
  python scripts/generation_sample_t2w.py $SAMPLE $COMMON;
fi
