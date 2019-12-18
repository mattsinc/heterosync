#!/bin/bash

NUM_RUNS=1
NUM_CS_ITERS=10
EXECUTABLES="allSyncPrims-1kernel"
SYNCPRIMS="atomicTreeBarrUniq atomicTreeBarrUniqLocalExch lfTreeBarrUniq lfTreeBarrUniqLocalExch spinMutex spinMutexEBO faMutex sleepMutex spinSem10 spinSemEBO10 spinMutexUniq spinMutexEBOUniq faMutexUniq sleepMutexUniq spinSemUniq10 spinSemEBOUniq10" # syncPrims to run
#NUM_LDST="10 100 1000"
NUM_LDST="10"
# This GPU has 2 SMs, so want 1 TB/SM, 2 TB/SM, 4 TB/SM, 8 TB/SM (max allowed)
NUM_TBS="2 4 8 16"

# do the prescribed number of runs for each executable, print out all runtimes
for executable in $EXECUTABLES;
do 
    echo "Beginning $executable's tests"

    for syncPrim in $SYNCPRIMS;
    do
        echo -e "\tsyncPrim = $syncPrim"

        for numLdSt in $NUM_LDST
        do
            echo -e "\t\tnumLdSt = $numLdSt"

            for numTBs in $NUM_TBS
            do
                echo -e "\t\t\tnumTBs = $numTBs"

                # ./allSyncPrims-1kernel <syncPrim> <numLdSt> <numTBs> <numCSIters>
                echo -e "\t\t\t\t./$executable $syncPrim $numLdSt $numTBs $NUM_CS_ITERS"
                for (( j=0; j<$NUM_RUNS; j++ ))
                do
                    #echo -e "\t\t\tRun $j"
                    duration[$j]=0
                    duration[$j]=`./$executable $syncPrim $numLdSt $numTBs $NUM_CS_ITERS | grep "average" | cut -f3 -d: | cut -f1 -dm`
                    #echo -e "\t\t\t\tduration = ${duration[$j]}"
                done

                # print out all runtimes, comma separated, for this executable
                echo -n -e "\t\t\t\t\tAll durations for $executable: "
                for (( j=0; j<$NUM_RUNS; j++ ))
                do
                    echo -n "${duration[$j]}, "
                done
                echo "" # newline

                # average the duration for these threads
                threadnum_duration=0
                for (( j=0; j<$NUM_RUNS; j++ ))
                do
                    #echo -e "\tBeginning $j""th run's write"
                    threadnum_duration=`echo $threadnum_duration"+"${duration[$j]}"/"$NUM_RUNS|bc -l`
                done
                echo -e "\t\t\t\tAverage = $threadnum_duration"
            done
        done
    done
done

