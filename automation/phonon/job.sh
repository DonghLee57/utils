#! /bin/bash
# VASP-based phonon calculation workflow
# INCAR_MOD_1, INCAR_MOD_2: They are required as they can override tags from the base INCAR file, even if they are empty documents.
# get_POTCAR.py: Code that generates a POTCAR from a POSCAR input

$CALC_VASP='command for running vasp'
MPIRUN='/your_mpirun'
PYTHON='/your_python'
PHONOPY='/your_phonopy'
INPUTS='/vasp_input/' # INCAR_RELAX: 
SAVE_MAT='../materials' 

JOB_ARRAY=(1 1) # (relax phonopy)
MAT=$(basename "$PWD")
JOBDIR=$(echo $PWD)
if [ "${JOB_ARRAY[0]}" -eq 1 ]; then
    WORKING_DIR_1="./relax"
    if [ ! -d "${WORKING_DIR_1}" ]; then
        mkdir "${WORKING_DIR_1}"
        echo "Directory '${WORKING_DIR_1}' created successfully."
    else
        echo "Directory '${WORKING_DIR_1}' already exists."
    fi

    cp  machines ${WORKING_DIR_1}/machines
    cp  ${SAVE_MAT}/POSCAR_${MAT}.vasp  ${WORKING_DIR_1}/POSCAR
    cat  INCAR_MOD_1  ${INPUTS}/INCAR_RELAX  >  ${WORKING_DIR_1}/INCAR
 
    MAX_LOOPS=5
    LOOPS=0   
    cd ${WORKING_DIR_1}
        $PYTHON ${INPUTS}/get_POTCAR.py POSCAR
        while true; do
            $CALC_VASP
            cp CONTCAR POSCAR
            TAIL_LAST=$(tail -n 1 OSZICAR)
            FNUMBER=$(echo "$TAIL_LAST" | grep -oP '\s+\d+(?=\s+F=)')
            if [[ "$FNUMBER" == *"1"* ]]; then
                echo "Stopping: OSZICAR shows convergence (F= line detected)"
                cd ${JOBDIR}
                break
            fi
            if [ $LOOPS -ge $MAX_LOOPS ]; then
                echo "Stopping: Reached maximum number of loops ($MAX_LOOPS)"
                cd ${JOBDIR}
                break
            fi
            ((LOOPS++))
        done
fi

if [ "${JOB_ARRAY[1]}" -eq 1 ]; then
    WORKING_DIR_2="./phonopy"
    if [ ! -d "${WORKING_DIR_2}" ]; then
        mkdir "${WORKING_DIR_2}"
        echo "Directory '${WORKING_DIR_2}' created successfully."
    else
        echo "Directory '${WORKING_DIR_2}' already exists."
    fi

    cd ${WORKING_DIR_2}
        cp ${JOBDIR}/${WORKING_DIR_1}/POSCAR ./POSCAR
        $PHONOPY -d --dim 2 2 2 --pa auto

        for file in POSCAR-*;
            do
                number=$(echo "$file" | sed 's/POSCAR-//')
                mkdir -p "$number"
                mv "$file" "$number/POSCAR"

                cp  ${JOBDIR}/machines ${number}/machines
                cat ${JOBDIR}/INCAR_MOD_2  ${INPUTS}/INCAR_RELAX  >  ${number}/INCAR
                cd ${number}
                  $PYTHON ${INPUTS}/get_POTCAR.py POSCAR
                  $CALC_VASP
                cd ${JOBDIR}/${WORKING_DIR_2}

                dir_list+=("$number")
            done

    phonopy_dirs=""
    for dir in "${dir_list[@]}"; do
        phonopy_dirs="${phonopy_dirs} ${dir}/vasprun.xml"
    done

    $PHONOPY -f ${phonopy_dirs}
    $PHONOPY --qpoints="0 0 0" --dim 2 2 2
fi
