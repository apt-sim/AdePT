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
    string(name: 'ghprbActualCommit', description: 'Immutable PR commit SHA validated by GitHub Actions')
    string(name: 'ghprbPullAuthorLogin', description: 'Author of the Pull Request (provided by GitHub)')
    string(name: 'ghprbPullId', description: 'Pull Request id (provided by GitHub)')
  }

  environment {
    CMAKE_SOURCE_DIR     = 'AdePT'
    CMAKE_BINARY_DIR     = 'build'
    CMAKE_INSTALL_PREFIX = 'install'
    CUDA_CAPABILITY      = '75'
    BUILDTYPE            = "${params.BUILDTYPE}"
  }

  agent none
  options {
    skipDefaultCheckout(true)
  }

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
        stage('CheckoutPR') {
          steps {
            checkoutPrSource()
          }
        }
        stage('CheckoutReference') {
          when {
            expression { params.ghprbPullId?.trim() }
          }
          steps {
            checkoutReferenceSource()
          }
        }
        stage('BuildPR') {
          steps {
            buildMatrixFromSource('build_pr_matrix', 'AdePT', 'BUILD')
          }
        }
        stage('BuildReference') {
          when {
            expression { params.ghprbPullId?.trim() }
          }
          steps {
            buildMatrixFromSource('build_master_reference_matrix', 'AdePT_master_reference', 'BUILD_MASTER_REFERENCE')
          }
        }
        stage('Test') {
          steps {
            runTestsForBuiltMatrices()
          }
        }
      }
      post {
        success {
          deleteDir()
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
        stage('CheckoutPR') {
          steps {
            checkoutPrSource()
          }
        }
        stage('CheckoutReference') {
          when {
            expression { params.ghprbPullId?.trim() }
          }
          steps {
            checkoutReferenceSource()
          }
        }
        stage('BuildPR') {
          steps {
            buildMatrixFromSource('build_pr_matrix', 'AdePT', 'BUILD')
          }
        }
        stage('BuildReference') {
          when {
            expression { params.ghprbPullId?.trim() }
          }
          steps {
            buildMatrixFromSource('build_master_reference_matrix', 'AdePT_master_reference', 'BUILD_MASTER_REFERENCE')
          }
        }
        stage('Test') {
          steps {
            runTestsForBuiltMatrices()
          }
        }
      }
      post {
        success {
          deleteDir()
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------------------------
//---Common Functions---------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------

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
    def dev_out = sh (script: deviceQuery, returnStdout: true)
    env.CUDA_CAPABILITY = ( dev_out =~ 'CUDA Capability.*([0-9]+[.][0-9]+)')[0][1].replace('.','')
    print('Cuda capability version is = ' + env.CUDA_CAPABILITY)
  }
}

def withLcgEnvScript(String body) {
  return """
    set +x
    source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-${OS}-${COMPILER}-opt/setup.sh
    set -x
    export CUDA_CAPABILITY=${env.CUDA_CAPABILITY}
    ${body}
  """
}

def runWithLcgEnv(String stepLabel, String body) {
  sh label: stepLabel, script: withLcgEnvScript(body)
}

def runWithLcgEnvStatus(String stepLabel, String body) {
  return sh(label: stepLabel, returnStatus: true, script: withLcgEnvScript(body))
}

def runLabeledCiTest(String stepLabel, String binaryDir, String includeLabel) {
  runWithLcgEnv(stepLabel, """
    export CMAKE_SOURCE_DIR="\$PWD/AdePT"
    export CMAKE_BINARY_DIR="${binaryDir}"
    export CTEST_INCLUDE_LABEL="${includeLabel}"
    ctest -V --output-on-failure -S "\$PWD/AdePT/jenkins/adept-ctest-ci.cmake,\$MODEL"
  """)
}

def runBuildMatrix(String stepLabel, String sourceDir, String buildPrefix) {
  runWithLcgEnv(stepLabel, """
    run_build_slot() {
      local source_dir=\$1
      local binary_dir=\$2
      shift 2
      local extra_opts="\$*"

      export CMAKE_SOURCE_DIR="\${source_dir}"
      export CMAKE_BINARY_DIR="\${binary_dir}"
      export ExtraCMakeOptions="-DADEPT_BUILD_TESTING=ON \${extra_opts}"
      ctest -V --output-on-failure -S "\$PWD/AdePT/jenkins/adept-ctest-build.cmake,\$MODEL"
    }

    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_MONOL"
    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_SPLIT_ON" "-DADEPT_USE_SPLIT_KERNELS=ON"
    run_build_slot "${sourceDir}" "\$PWD/${buildPrefix}_MIXED_PRECISION" "-DADEPT_MIXED_PRECISION=ON"
  """)
}

def checkoutPrSource() {
  // `checkout scm` already carries the job-configured target dir (AdePT),
  // so run it at workspace root to avoid nesting into AdePT/AdePT.
  deleteDir()
  checkout scm
  dir('AdePT') {
    sh 'git submodule update --init'
    if (params.ghprbActualCommit?.trim()) {
      sh """
        actual_commit=\$(git rev-parse HEAD)
        expected_commit="${params.ghprbActualCommit}"
        echo "Checked out PR commit: \${actual_commit}"
        echo "Expected PR commit:   \${expected_commit}"
        if [ "\${actual_commit}" != "\${expected_commit}" ]; then
          echo "ERROR: Jenkins checked out a different PR commit than the one validated by GitHub Actions." >&2
          exit 1
        fi
      """
    }
  }
}

def checkoutReferenceSource() {
  def remoteConfig = [url: scm.userRemoteConfigs[0].url]
  if (scm.userRemoteConfigs[0].credentialsId) {
    remoteConfig.credentialsId = scm.userRemoteConfigs[0].credentialsId
  }

  dir('AdePT_master_reference') {
    deleteDir()
    checkout([
      $class: 'GitSCM',
      branches: [[name: '*/master']],
      doGenerateSubmoduleConfigurations: false,
      userRemoteConfigs: [remoteConfig]
    ])
    sh 'git submodule update --init'
  }
}

def buildMatrixFromSource(String stepLabel, String sourceSubdir, String buildPrefix) {
  runBuildMatrix(stepLabel, "\$PWD/${sourceSubdir}", buildPrefix)
}

def runTestsForBuiltMatrices() {
  boolean isPrBuild = params.ghprbPullId?.trim()
  boolean runValidationTests = true

  if (isPrBuild) {
    def driftStatus = runWithLcgEnvStatus('physics_drift', """
      bash "\$PWD/AdePT/jenkins/run_physics_drift_tests.sh" \\
           "\$PWD" \\
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
