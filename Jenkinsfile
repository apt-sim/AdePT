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
    choice(name: 'COMPILER', choices: ['gcc8', 'gcc10', 'clang10', 'native'])
    choice(name: 'BUILDTYPE', choices: ['Release', 'Debug'])
    string(name: 'LABEL', defaultValue: 'TeslaT4', description: 'Jenkins label for physical nodes or container image for docker')
    string(name: 'ExtraCMakeOptions', defaultValue: '', description: 'CMake extra configuration options')
    string(name: 'DOCKER_LABEL', defaultValue: 'docker-host-noafs', description: 'Label for the the nodes able to launch docker images')
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
        label "$LABEL"
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
  }
}

def setJobName() {
  currentBuild.displayName = "#${BUILD_NUMBER}" + ' ' + params.COMPILER + '-' + params.BUILDTYPE
}

def buildAndTest() {
  sh label: 'build_and_test', script: """
    source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/x86_64-centos7-${COMPILER}-opt/setup.sh
    env | sort | sed 's/:/:?     /g' | tr '?' '\n'
    ctest -VV -S AdePT/jenkins/adept-ctest.cmake,$MODEL
  """
}