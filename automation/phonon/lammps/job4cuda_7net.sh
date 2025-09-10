NGPU='1'
export CUDA_VISIBLE_DEVICES=0,1

### User setting ###
SCRIPT='script.in'
POT_SERIAL='/your_pot.pt'
POT_PARA='/your_pot/'
FILE_COUNT=$(find "$POT_PARA" -maxdepth 1 -type f | wc -l)

export TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0;8.6;8.9;9.0"
export FUSED=1

export OMP_NUM_THREADS=32

### MPIRUN & LAMMPS path ###
MPIRUN='your_mpirun'
LMPSVN='your_lammps'
PYTHON='sevennet_python'
PHONOPY='phonopy_python'
SAVE_MAT='../../materials'
SUPERCELL_DIM="2 2 2"

JOB_ARRAY=(1 1) # (relax phonopy)
MAT=$(basename "$PWD")
JOBDIR=$(echo $PWD)
WORKING_DIR_1="./relax"
WORKING_DIR_2="./phonopy"
if [ "${JOB_ARRAY[0]}" -eq 1 ]; then
    if [ ! -d "${WORKING_DIR_1}" ]; then
        mkdir "${WORKING_DIR_1}"
        echo "Directory '${WORKING_DIR_1}' created successfully."
    else
        echo "Directory '${WORKING_DIR_1}' already exists."
    fi

    ELEM=$($PYTHON vasp2lammps.py ${SAVE_MAT}/POSCAR_${MAT}.vasp ${WORKING_DIR_1})
 
    $PYTHON set_coeff_mass.py serial $ELEM >  pot.info
    cat script_header.in pot.info script_relax.in > ${WORKING_DIR_1}/${SCRIPT}

    cd ${WORKING_DIR_1}
        $LMPSVN -in script.in -log lammps.log -var POT ${POT_SERIAL} -var RND '2025'\
                                              -var input_structure 'unit.lammps'
        cd ${JOBDIR}
fi

if [ "${JOB_ARRAY[1]}" -eq 1 ]; then
    if [ ! -d "${WORKING_DIR_2}" ]; then
        mkdir "${WORKING_DIR_2}"
        echo "Directory '${WORKING_DIR_2}' created successfully."
    else
        echo "Directory '${WORKING_DIR_2}' already exists."
    fi

    ELEM=$($PYTHON vasp2lammps.py ${SAVE_MAT}/POSCAR_${MAT}.vasp)
    $PYTHON lammps2lammps.py ${WORKING_DIR_1}/relaxed.lammps ${WORKING_DIR_2} ${ELEM}
    cd ${WORKING_DIR_2}
        $PHONOPY -d --dim ${SUPERCELL_DIM} --pa auto  --lammps -c unit.lammps

        for file in supercell-*;
            do
                number=$(echo "$file" | sed 's/supercell-//')
                mkdir -p "$number"
                mv "$file" "${number}/supercell"

                cat ${JOBDIR}/{script_header.in,pot.info,script_phonopy.in} > ${JOBDIR}/${WORKING_DIR_2}/${number}/${SCRIPT}

                cd ${number}
                    $LMPSVN -in script.in -log lammps.log -var POT ${POT_SERIAL} -var RND '2025'\
                                                          -var input_structure 'supercell'
                    #$MPIRUN -n $NGPU $LMPSVN -in script.in -log lammps.log -var POT ${POT_SERIAL} -var NMPL ${FILE_COUNT} -var RND '2025'
                cd ${JOBDIR}/${WORKING_DIR_2}

                dir_list+=("$number")
            done

    phonopy_dirs=""
    for dir in "${dir_list[@]}"; do
        phonopy_dirs="${phonopy_dirs} ${dir}/force.dump"
    done

    $PHONOPY -f ${phonopy_dirs}
    $PHONOPY --qpoints="0 0 0" --dim ${SUPERCELL_DIM}\
             --pa 1 0 0 0 1 0 0 0 1
fi
