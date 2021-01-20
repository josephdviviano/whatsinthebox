#!/bin/bash
START=1
END=720

for i in $(seq ${START} ${END}); do
    filenae="witb_${i}"
    runscript="${filename}.pbs"
    cat << EOF > ${runscript}

#!/bin/bash
#SBATCH --job-name=${filename}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=../outputs/${filename}.log
#SBATCH --error=../outputs/${filename}.err
#SBATCH --time=6:00:00
#SBATCH --mem=16Gb
#SBATCH --gres=gpu:1

hostname
source $HOME/.bashrc
source activate perplex
echo "running on shard ${i}/${END}"

python -u ../witb/main.py --remote=https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/wet.paths.gz --idx=${i} --output=../outputs/${filename}.pkl

EOF

    srun ${runscript}
    rm ${runscript}
done

