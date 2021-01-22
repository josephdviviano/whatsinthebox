#!/bin/bash
i=0
DATA_PATH="/home/sashalu/CCdata/"
for f in $(ls ${DATA_PATH}); do
	input="${DATA_PATH}/${f}"
    filename="witb_${i}"
    runscript="${filename}.pbs"
    cat << EOF > ${runscript}
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
echo "running on shard ${input}"

python -u ../witb/main.py --file=${input} --output=../outputs/${filename}.pkl

EOF

	i=$((i+1))  # this iterates a counter
    sbatch ${runscript}
    rm ${runscript}
done

