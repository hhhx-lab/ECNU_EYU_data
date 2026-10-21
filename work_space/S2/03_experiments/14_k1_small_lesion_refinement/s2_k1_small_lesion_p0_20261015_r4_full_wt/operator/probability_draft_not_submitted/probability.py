"""Frozen two-checkpoint inference; bounded staging, no worker queue or training."""
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from cpu_roundtrip import ROOT, emit, require, sha


def check():
    p=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
    for name,digest in p['static_sha256'].items():require(sha(ROOT/name)==digest,'static drift '+name)
    for epoch,record in p['probability']['checkpoints'].items():
        require(sha(Path(record['path']))==record['sha256'],'probability checkpoint SHA drift '+epoch)
    return p


def predictor(p,epoch,device):
    import torch
    import nnunetv2.inference.predict_from_raw_data as module
    from custom_nnunet.nnUNetTrainerBraTS2026RC_inference import nnUNetTrainerBraTS2026RC
    allowed='nnUNetTrainerBraTS2026RCMetAugK1245FocalCompletionFineTune175'
    def finder(name):
        require(name==allowed,'unexpected checkpoint trainer')
        return nnUNetTrainerBraTS2026RC
    module.recursive_find_trainer_class_by_name=finder
    pred=module.nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,
        perform_everything_on_device=device.type=='cuda',device=device,verbose=False,
        verbose_preprocessing=False,allow_tqdm=False)
    path=Path(p['probability']['checkpoints'][str(epoch)]['path'])
    pred.initialize_from_trained_model_folder(str(path.parents[1]),use_folds=(0,),checkpoint_name=path.name)
    require(pred.trainer_name==allowed,'trainer mismatch')
    require(sha(path.parents[1]/'plans.json')==p['dependency_sha256'][p['plans_path']],'checkpoint plans drift')
    require(sha(path.parents[1]/'dataset.json')==p['dependency_sha256'][p['dataset_path']],'checkpoint dataset drift')
    pred.network.to(device).eval()
    return pred


def preflight():
    import torch
    import numpy as np
    p=check()
    require(not torch.cuda.is_available(),'preflight must hide GPU')
    for epoch in (175,150):
        pred=predictor(p,epoch,torch.device('cpu'))
        require(all(torch.isfinite(v).all() for v in pred.network.state_dict().values()),'nonfinite network')
        import inspect
        require('dataset_json' in inspect.signature(pred.configuration_manager.preprocessor_class.run_case_npy).parameters,'preprocessor API drift')
        del pred
    from transport_rules import working_point
    probs=np.zeros((5,2,2,2),dtype=np.float32);probs[0]=1
    baseline=np.zeros((2,2,2),dtype=np.uint8);baseline[0,0,0]=4
    for threshold in p['probability']['wt_thresholds']:
        require(np.array_equal(working_point(probs,threshold,baseline)==4,baseline==4),'RC protection drift')
    emit(ROOT/'evidence/PROBABILITY_STATIC_PREFLIGHT.json',dict(status='pass',both_networks_loaded_cpu=True,
        source_and_checkpoint_sha=True,preprocessor_signature=True,scope='not_GPU_prediction_or_full_probability_acceptance'))


def run():
    import numpy as np
    import torch
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    p=check();q=p['probability'];batch=q['batches'][0]
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'exactly one visible GPU')
    require('A100' in torch.cuda.get_device_name(0),'A100 required')
    free=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
    needed=q['other_G1_full_reserve_bytes']+q['cpu_reserve_bytes']+batch['stage_budget_bytes']
    if free<needed:
        emit(ROOT/'evidence/PROBABILITY_CAPACITY_BLOCKED.json',dict(status='blocked_before_inference',available_bytes=free,needed_bytes=needed))
        return
    out=ROOT/'probability_stage/batch_000';out.mkdir(parents=True,exist_ok=False)
    emit(ROOT/'evidence/PROBABILITY_STARTED.json',dict(status='started',canonical_job_id=os.environ.get('SLURM_JOB_ID'),
        device_name=torch.cuda.get_device_name(0),batch_index=0,cases=batch['cases'],training=False,
        preprocess_workers=0,export_workers=0,multiprocessing_queue_used=False))
    geometry={r['subject_id']:r for r in q['geometry']}
    records=[]
    for epoch in (175,150):
        pred=predictor(p,epoch,torch.device('cuda',0))
        preprocessor=pred.configuration_manager.preprocessor_class(verbose=False)
        rw=pred.plans_manager.image_reader_writer_class()
        for case in batch['cases']:
            inputs=[str(Path(p['cohort_root'])/'input'/(case+'_%04d.nii.gz'%i)) for i in range(4)]
            require(all(sha(Path(f))==p['input_reference_sha256'][f] for f in inputs),'input drift')
            image,props=rw.read_images(inputs)
            data,_,properties=preprocessor.run_case_npy(image,None,copy.deepcopy(props),pred.plans_manager,
                pred.configuration_manager,pred.dataset_json)
            require(list(data.shape[1:])==geometry[case]['training_shape'],'training geometry drift')
            logits=pred.predict_logits_from_preprocessed_data(torch.from_numpy(np.ascontiguousarray(data))).float().cpu()
            train=pred.label_manager.apply_inference_nonlin(logits).cpu().numpy().astype(np.float32,copy=False)
            segmentation,original=convert_predicted_logits_to_segmentation_with_correct_shape(logits,
                pred.plans_manager,pred.configuration_manager,pred.label_manager,properties,True,num_threads_torch=1)
            original=np.asarray(original,dtype=np.float32)
            require(list(original.shape[1:])==geometry[case]['original_shape'],'original geometry drift')
            for array in (train,original):
                require(array.dtype==np.float32 and array.shape[0]==5 and np.isfinite(array).all(),'probability dtype/finite')
                require(array.min()>=0 and array.max()<=1 and np.allclose(array.sum(0),1,atol=2e-4,rtol=0),'probability range/sum')
            require(set(np.unique(segmentation))<=set(range(5)),'illegal prediction labels')
            if epoch==175:
                from supplement import image as read_label
                baseline,_=read_label(Path(p['prediction_roots']['K1'])/(case+'.nii.gz'),p['prediction_sha256']['K1'][case+'.nii.gz'])
                require(np.array_equal(segmentation,baseline),'175 baseline prediction voxel drift; no scan authorized')
            target=out/str(epoch)/case;target.mkdir(parents=True,exist_ok=False)
            for name,array in (('training_probabilities.npy',train),('original_probabilities.npy',original),('argmax.npy',segmentation.astype(np.uint8))):
                with (target/name).open('xb') as f:np.save(f,array,allow_pickle=False)
            emit(target/'GEOMETRY_AND_CHECKPOINT.json',dict(status='pass',case=case,epoch=epoch,
                checkpoint_sha256=q['checkpoints'][str(epoch)]['sha256'],training_shape=list(train.shape),
                original_shape=list(original.shape),dtype='float32',properties=properties,
                baseline175_voxelwise_verified=epoch==175,full_p0_complete=False))
            records.append(dict(case=case,epoch=epoch,path=str(target.relative_to(ROOT))))
            print('PROBABILITY_CASE',epoch,case,flush=True)
            del image,data,logits,train,original,segmentation
        del pred;torch.cuda.empty_cache()
    files={str(f.relative_to(ROOT)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in out.rglob('*') if f.is_file()}
    require(sum(f['bytes'] for f in files.values())<=batch['stage_budget_bytes'],'stage budget exceeded')
    emit(ROOT/'evidence/PROBABILITY_BATCH_VALIDATION.json',dict(status='pass_batch_only',batch_index=0,
        case_count=len(batch['cases']),epochs=[175,150],records=records,files=files,
        full_cohort_cases=103,full_probability_complete=False,needs_verified_local_pull_before_cleanup=True))
    with (ROOT/'evidence/PROBABILITY_BATCH_COMPLETE.ok').open('x') as f:f.write('pass_batch_only\n')


if __name__=='__main__':
    try:
        if sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='run':run()
        else:raise ValueError('unknown action')
    except Exception:
        emit(ROOT/'evidence/PROBABILITY_FAILED.json',dict(status='failed',error=traceback.format_exc()))
        raise
