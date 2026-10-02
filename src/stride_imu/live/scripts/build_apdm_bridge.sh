#!/bin/sh
# Compile bridge/ApdmBridge.java against APDM's apdm.jar (shipped inside
# Motion Studio) and run a probe with Motion Studio's bundled x86_64 JRE.
# Needs any JDK's javac (targets Java 8 so the bundled JRE 8 can run it).
set -e
cd "$(dirname "$0")/.."
SDK="/Applications/MotionStudio.app/Contents/Resources/configuration/org.eclipse.osgi/6/0/.cp/apdm_sdk"
JAVA="/Applications/MotionStudio.app/Contents/Resources/jre/Contents/Home/jre/bin/java"
mkdir -p bridge/build
javac --release 8 -nowarn -cp "$SDK/java/apdm.jar" -d bridge/build bridge/ApdmBridge.java
echo "built bridge/build/ApdmBridge.class"
"$JAVA" -Djava.library.path="$SDK/libs/MacOSX/x64" -cp "bridge/build:$SDK/java/apdm.jar" ApdmBridge probe
