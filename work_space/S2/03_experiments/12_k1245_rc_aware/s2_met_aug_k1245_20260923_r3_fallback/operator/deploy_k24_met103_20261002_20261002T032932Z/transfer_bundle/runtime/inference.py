"""Run the preserved sequential inference engine for the exact K1245 trainer."""
import frozen_s2_inference as frozen

frozen.ALLOWED_TRAINERS = {'nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'}

if __name__ == '__main__':
    frozen.main()
