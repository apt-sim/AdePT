//----------------------------------------------------------------------------------------------------------------------
// This declarative Jenkins pipeline encodes all the steps required for the nightly/continuous of a single platform.
// Other jobs may call this pipeline to execute the build, test and installation of a set platforms.
//
// Author: Pere Mato
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//----------------------------------------------------------------------------------------------------------------------

pipeline {
  parameters {
    string(name: 'EXTERNALS', defaultValue: 'devAdePT/latest', description: 'LCG software stack in CVMFS')
    choice(name: 'MODEL', choices: ['experimental', 'nightly', 'continuous'], description: 'CDash model')
    choice(name: 'COMPILER', choices: ['gcc13', 'gcc11', 'gcc8', 'gcc10', 'clang10', 'native'])
    choice(name: 'OS', choices: ['el9', 'centos7'])
    choice(name: 'BUILDTYPE', choices: ['Release', 'Debug'])
    string(name: 'LABEL', defaultValue: 'TeslaT4', description: 'Jenkins label for physical nodes or container image for docker')
    string(name: 'ExtraCMakeOptions', defaultValue: '', description: 'CMake extra configuration options')
    string(name: 'DOCKER_LABEL', defaultValue: 'docker-host-noafs', description: 'Label for the the nodes able to launch docker images')
    string(name: 'ghprbPullAuthorLogin', description: 'Author of the Pull Request (provided by GitHub)')
    string(name: 'ghprbPullId', description: 'Pull Request id (provided by GitHub)')
  }

  environment {
    CMAKE_SOURCE_DIR     = 'AdePT'
    CMAKE_BINARY_DIR     = 'build'
    CMAKE_INSTALL_PREFIX = 'install'
  }

  agent none

  stages {
    //------------------------------------------------------------------------------------------------------------------
    //---Build & Test stages--------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    stage('Environment'){
      steps {
        setJobName()
      }
    }
    stage('InDocker') {
      when {
        beforeAgent true
        expression { params.LABEL =~ 'centos|ubuntu' }
      }
      agent {
        docker {
          image "gitlab-registry.cern.ch/sft/docker/$LABEL"
          label "$DOCKER_LABEL"
          args  """-v /cvmfs:/cvmfs
                   -v /ec:/ec
                   -e SHELL 
                   -e ghprbPullAuthorLogin 
                   -e ghprbPullId
                   --net=host
                   --hostname ${LABEL}-docker
                """
        }
      }
      stages {
        stage('Build&Test') {
          steps {
            buildAndTest()
          }
          post {
            success {
              deleteDir()
            }
          }
        }
      }
    }
    stage('InBareMetal') {
      when {
        beforeAgent true
        expression { params.LABEL =~ 'cuda|physical|TeslaT4' }
      }
      agent {
        label "$LABEL-$OS"
      }
      stages {
        stage('PreCheckNode') {
          steps {
            preCheckNode()
          }
        }
        stage('Build&Test') {
          steps {
            buildAndTest()
          }
          post {
            success {
              deleteDir()
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
//---Common Functions---------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

def CUDA_CAPABILITY = '75'  // Default is 7.5

def setJobName() {
  if (params.ghprbPullId) {
    currentBuild.displayName = "#${BUILD_NUMBER}" + '-' + params.ghprbPullAuthorLogin + '#' +  params.ghprbPullId + '-' + params.COMPILER + '-' + params.BUILDTYPE
  } 
  else {
    currentBuild.displayName = "#${BUILD_NUMBER}" + ' ' + params.COMPILER + '-' + params.BUILDTYPE
  }
}

def preCheckNode() {
  def deviceQuery = '/usr/local/cuda/extras/demo_suite/deviceQuery'
  sh (script: '/usr/bin/nvidia-smi')
  if (fileExists(deviceQuery)) {
    dev_out = sh (script: deviceQuery, returnStdout: true)
    CUDA_CAPABILITY = ( dev_out =~ 'CUDA Capability.*([0-9]+[.][0-9]+)')[0][1].replace('.','')
    print('Cuda capability version is = ' + CUDA_CAPABILITY)
  }
}

def runLabeledCiTest(String stepLabel, String binaryDir, String includeLabel) {
  sh label: stepLabel, script: """
    set +x
    source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-${OS}-${COMPILER}-opt/setup.sh
    set -x
    export CUDA_CAPABILITY=${CUDA_CAPABILITY}

    export CMAKE_SOURCE_DIR="\$PWD/AdePT"
    export CMAKE_BINARY_DIR="${binaryDir}"
    export CTEST_INCLUDE_LABEL="${includeLabel}"
    ctest -V --output-on-failure --timeout 2400 -S "\$PWD/AdePT/jenkins/adept-ctest-ci.cmake,\$MODEL"
  """
}

def runBuildMatrix(String stepLabel, String sourceDir, String buildPrefix) {
  sh label: stepLabel, script: """
    set +x
    source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-${OS}-${COMPILER}-opt/setup.sh
    set -x
    export CUDA_CAPABILITY=${CUDA_CAPABILITY}

    run_build_slot() {
      local source_dir=\$1
      local binary_dir=\$2
      shift 2
      local extra_opts="\$*"

      export CMAKE_SOURCE_DIR="\${source_dir}"
      export CMAKE_BINARY_DIR="\${binary_dir}"
      export ExtraCMakeOptions="-DADEPT_BUILD_TESTING=ON \${extra_opts}"
      ctest -V --output-on-failure --timeout 2400 -S "\$PWD/AdePT/jenkins/adept-ctest-build.cmake,\$MODEL"
    }

    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_MONOL"
    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_SPLIT_ON" "-DADEPT_USE_SPLIT_KERNELS=ON"
    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_MIXED_PRECISION" "-DADEPT_MIXED_PRECISION=ON"
  """
}

def buildAndTest() {
  dir('AdePT') {
    sh 'git submodule update --init'
  }
  boolean isPrBuild = params.ghprbPullId?.trim()
  boolean runValidationTests = true

  runBuildMatrix('build_pr_matrix', '$PWD/AdePT', 'BUILD')

  if (isPrBuild) {
    sh label: 'prepare_master_reference', script: """
      set +x
      source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-${OS}-${COMPILER}-opt/setup.sh
      set -x

      git -C "\$PWD/AdePT" worktree remove --force "\$PWD/AdePT_master_reference" >/dev/null 2>&1 || true
      rm -rf "\$PWD/AdePT_master_reference"

      git -C "\$PWD/AdePT" fetch --no-tags origin +refs/heads/master:refs/remotes/origin/master
      git -C "\$PWD/AdePT" worktree add --force "\$PWD/AdePT_master_reference" origin/master
      git -C "\$PWD/AdePT_master_reference" submodule update --init
    """

    runBuildMatrix('build_master_reference_matrix', '$PWD/AdePT_master_reference', 'BUILD_MASTER_REFERENCE')

    def driftStatus = sh(label: 'physics_drift', returnStatus: true, script: """
      set +x
      source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-${OS}-${COMPILER}-opt/setup.sh
      set -x
      export CUDA_CAPABILITY=${CUDA_CAPABILITY}
      bash "\$PWD/AdePT/jenkins/master_vs_pr_validation.sh" \\
           "\$PWD" \\
           "\$PWD/AdePT" \\
           "\$PWD/AdePT_master_reference" \\
           "\$MODEL"
    """)

    if (driftStatus != 0) {
      unstable('physics_drift differences detected against master (non-blocking)')
      runValidationTests = true
    } else if (params.MODEL == 'nightly') {
      runValidationTests = true
      echo 'physics_drift passed, but this is a nightly model: running validation.'
    } else {
      runValidationTests = false
      echo 'physics_drift passed: skipping validation tests for this PR build.'
    }
  }

  runLabeledCiTest('run_unit_tests', '$PWD/BUILD_MONOL', 'unit')

  if (runValidationTests) {
    runLabeledCiTest('run_validation_monol', '$PWD/BUILD_MONOL', 'validation')
    runLabeledCiTest('run_validation_split', '$PWD/BUILD_SPLIT_ON', 'validation')
  } else {
    echo 'Validation tests skipped because PR physics drift matched master exactly.'
  }
}
