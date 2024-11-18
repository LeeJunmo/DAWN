# Our best performance hyperparameter set for each dataset.

dcrs=("48hrs")
early_thres=("0.3")
samples=("1000")
margins=("0.1")
alphas=("0.1")

for dcr in "${dcrs[@]}"; do
    for e_thrs in "${early_thres[@]}"; do
        for sample in "${samples[@]}"; do
            for mg in "${margins[@]}"; do
                for alpha in "${alphas[@]}"; do
                        python run.py --dataset_name 'politifact' --gpu 5 --epochs 1000 --iters 5 --dead_criterion "$dcr" --early_thres "$e_thrs" --sampling_num "$sample" --margin "$mg" --alpha "$alpha" --module 'gcn_ef'
                done
            done
        done
    done
done

dcrs=("12hrs")
early_thres=("0.7")
samples=("10000")
margins=("0.0")
alphas=("0.3")

for dcr in "${dcrs[@]}"; do
    for e_thrs in "${early_thres[@]}"; do
        for sample in "${samples[@]}"; do
            for mg in "${margins[@]}"; do
                for alpha in "${alphas[@]}"; do
                    python run.py --dataset_name 'gossipcop' --gpu 5 --epochs 1000 --iters 5 --dead_criterion "$dcr" --early_thres "$e_thrs" --sampling_num "$sample" --margin "$mg" --alpha "$alpha" --module 'gcn_ef'
                done
            done
        done
    done
done
