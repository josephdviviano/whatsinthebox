#!/bin/bash
i=0
for f in $(ls /path/to/files/); do 
	input=${f}
	output=path/to/output/dir/witb_${i}.pkl
	i=$((i+1))  # this iterates a counter
	done

#!/bin/bash
#SBATCH --job-name=witb_${i}
#SBATCH --nodes=1-1
#SBATCH --ntasks-per-node=8
#SBATCH --output=../outputs/witb_${i}.log
#SBATCH --error=../outputs/with${i}.err
#SBATCH --time=6:00:00
#SBATCH --mem=16Gb
#SBATCH --gres=gpu:1
#SBATCH --account=rrg-bengioy-ad

hostname
source $HOME/.bashrc
source activate perplex
echo "running on shard ${i}/${END}"

python -u ../witb/main.py --file=/home/sashalu/CCdata/ --output=../outputs/witb_${i}.pkl

EOF

    sbatch ${runscript}
    rm ${runscript}
done

