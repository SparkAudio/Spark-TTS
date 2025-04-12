# Copyright (c) 2025 Abao (ibaoger@hotmail.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Parse command-line options.
# To be sourced by another script (as in ". parse_options.ps1").
# Option format is: --option-name arg
# and PowerShell variable "$option_name" gets set to value "arg."
# The exception is --help, which takes no arguments, but prints the
# $help_message variable (if defined).


# Process command line arguments
while ($args.Length -gt 0) {
    $arg = $args[0]
    
    # Handle --help option
    if ($arg -eq "--help" -or $arg -eq "-h") {
        if ([string]::IsNullOrEmpty($help_message)) {
            Write-Host "No help found." -ForegroundColor Red
        } else {
            Write-Host $help_message
        }
        exit 0
    }
    
    # Handle other options
    if ($arg -match "^--") {
        if ($arg -match "=") {
            Write-Host "Options to scripts must be of the form --name value, got '$arg'" -ForegroundColor Red
            exit 1
        }
        
        # Extract variable name from argument
        $name = $arg -replace "^--", "" -replace "-", "_"
        
        # Check if we have another argument as value
        if ($args.Length -lt 2) {
            Write-Host "Missing value for option $arg" -ForegroundColor Red
            exit 1
        }
        
        # Get the value and update arguments array
        $value = $args[1]
        $args = $args[2..($args.Length-1)]
        
        # Set the variable in the parent scope
        Set-Variable -Name $name -Value $value -Scope 1
        
    } else {
        # Not an option, move to next argument
        $args = $args[1..($args.Length-1)]
    }
}