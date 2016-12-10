#!/usr/bin/env python
"""
Created on Sat Sep 10 10:28:53 2016

@author: torres
"""

import subprocess
import platform
#import logging, logging.config

class DeIdentify(object):
    def __init__(self, input_file, hierarchy_folder='', logConf='logging.conf'):
        self.file = input_file
        self.hierarchy_folder = hierarchy_folder
        self.desensitize_action = 'none'
        self.desensitizedFiles = []
        #logging.config.fileConfig(logConf)
        #self.logger = logging.getLogger('autoML')

    def deIdentify(self, desensitize_action):

        self.desensitizedFiles = []
        
        if desensitize_action.lower() == 'manual':
            self.deIdentify_manual()
        elif desensitize_action.lower() == 'auto':
            self.deIdentify_auto()
        elif desensitize_action.lower() != 'none':
            print ("Invalid de-sensitization action: Changing to none")
            self.desensitize_action = 'none'
        
        if len(self.desensitizedFiles) == 0:   # no data privatizing chosen within ARX
            self.desensitizedFiles = [self.file]
            self.desensitize_action = 'none'
            

    def deIdentify_manual(self):
        """Perform manual de-identification/de-sensitization using ARX"""

        self.desensitize_action = 'manual'
        
        ver = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
        ver64 = '64-bit' in str( ver ).lower()
        platform_system = platform.system().lower()
    
        if 'win' in platform_system:
            if ver64:
                subprocess.run( ['java', '-jar', 'arx_autoML_windows64.jar', self.hierarchy_folder, self.file] )
            else:
                subprocess.run( ['java', '-jar', 'arx_autoML_windows32.jar', self.hierarchy_folder, self.file] )
        elif 'linux' in  platform_system:
            if ver64:
                subprocess.run( ['java', '-jar', 'arx_autoML_linux64.jar', self.hierarchy_folder, self.file] )
            else:
                subprocess.run( ['java', '-jar', 'arx_autoML_linux32.jar', self.hierarchy_folder, self.file] )
        elif 'darwin' in platform_system:
            if ver64:
                subprocess.run( ['java', '-jar', 'arx_autoML_cocoa_macOSx_x86_64.jar', self.hierarchy_folder, self.file] )
            else:
                subprocess.run( ['java', '-jar', 'arx_autoML_cocoa_macOSx.jar', self.hierarchy_folder, self.file] )
        else:
            raise SystemExit('Unsupported OS: ' + platform_system + ' is not supported')
            
        with open( self.hierarchy_folder + '/output.txt', 'r' ) as f:
            self.desensitizedFiles = [ line.strip() for line in f ]
        

    def deIdentify_auto(self):
        """Perform automatic privatization/de-sensitization using ARX"""
        self.desensitize_action = 'auto'
        raise SystemExit('auto mode for privatizing data not yet implemented')
        #  auto deidentifying is too difficult at this time
        

    def __str__(self):
        return(
        """File needing de-identification: %s
        de-sensitization mode: %s
        desensitized files: %s
        """ % ( self.file, self.desensitize_action, self.desensitizedFiles )
        )

#########
if __name__ == '__main__':
    g = DeIdentify("example.csv")
    g.deIdentify('manual')
    print ( g )

