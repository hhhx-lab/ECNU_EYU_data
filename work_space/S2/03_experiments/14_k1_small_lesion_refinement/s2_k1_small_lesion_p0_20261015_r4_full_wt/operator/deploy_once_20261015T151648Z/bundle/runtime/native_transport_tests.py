"""Exercise production enumeration through installed nnU-Net resampling/export."""
import copy
import hashlib
import json
from cpu_roundtrip import ROOT, require
from full_wt_transport import trace_volume


def run():
    import numpy as np
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    p=json.loads((ROOT/'CPU_EXECUTION_PLAN.json').read_text())
    pm=PlansManager(p['plans_path']);cm=pm.get_configuration('3d_fullres')
    dataset=json.loads(open(p['dataset_path']).read())

    def apply(ref,spacing,bbox=None):
        shape=[ref.shape[i] for i in pm.transpose_forward]
        bbox=bbox or [[0,x] for x in shape]
        sp=[spacing[i] for i in pm.transpose_forward]
        cropped=[b-a for a,b in bbox]
        target=compute_new_shape(cropped,sp,cm.spacing)
        props=dict(spacing=spacing,shape_before_cropping=shape,bbox_used_for_cropping=bbox,
            shape_after_cropping_and_before_resampling=cropped)
        export=lambda scores:convert_predicted_logits_to_segmentation_with_correct_shape(
            scores,pm,cm,pm.get_label_manager(dataset),copy.deepcopy(props),False,num_threads_torch=1)
        geom=dict(shape=list(ref.shape),spacing=list(reversed(spacing)),origin=[0,0,0],direction=[1,0,0,0,1,0,0,0,1])
        return trace_volume(ref,geom,'synthetic_native_boundary',hashlib.sha256(ref.tobytes()).hexdigest(),
            pm.transpose_forward,tuple(slice(a,b) for a,b in bbox),target,sp,cm.spacing,cm.resampling_fn_seg,export)

    zero=np.zeros((20,20,20),dtype=np.uint8)
    rows,targets,out=apply(zero,[1.,1.,1.]);require(not rows and not targets and not out.any(),'empty native trace')
    rc=zero.copy();rc[8:11,8:11,8:11]=4
    rows,targets,out=apply(rc,[1.,1.,1.]);require(not rows and not targets and np.array_equal(out,rc),'RC exclusion/label4 identity')
    boundary=zero.copy();boundary[3:6,3:6,3:6]=1;boundary[12,12,12]=3
    rows,_,out=apply(boundary,[1.,1.,1.])
    require(len(rows)==2 and sorted(r['original_volume_mm3'] for r in rows)==[1.,27.],'physical volume boundary')
    require(sum(r['small'] for r in rows)==1 and np.array_equal(out,boundary),'<27 not <=27')
    rows,_,_=apply(boundary,[1.,1.,1.],[[0,10]]*3)
    require(sum(r['transport_category']=='cropped_out' for r in rows)==1,'crop source classified before selection')
    # Deterministic synthetic native-input search; never label it as an observed patient.
    merge=None;attempts=0
    for spacing in (.6,.7,.8,.9):
        for start in (3,4,5):
            for gap in (1,2):
                ref=zero.copy();ref[start:start+7,5:12,5:12]=1
                x=start+7+gap;ref[x:x+2,7:9,7:9]=3
                rows,targets,out=apply(ref,[spacing]*3);attempts+=1
                if (len(rows)==2 and sum(r['small'] for r in rows)==1
                        and any(r['small'] and r['transport_category']=='merged' for r in rows)):
                    require(any(t['source_degree']==2 and t['small_source_count']==1 for t in targets),'full merge adjacency')
                    merge=dict(spacing=spacing,large_start=start,gap=gap,source_count=2,
                        small_sources=1,source_degree=2,source_volumes_mm3=[r['original_volume_mm3'] for r in rows],
                        provenance='constructed_synthetic_input_through_actual_installed_nnunet_pipeline')
                    break
            if merge:break
        if merge:break
    require(merge is not None,'actual native small-large merge boundary not exercised; stop')
    return dict(status='pass',empty=True,label4=True,physical_boundary_27=True,crop=True,
        actual_small_large_merge=merge,search_attempts=attempts,production_function='full_wt_transport.trace_volume')
