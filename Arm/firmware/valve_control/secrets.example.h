// Wi-Fi settings of the valve board. Copy this file to secrets.h (ignored by Git) and
// fill it in; never commit real names or passwords.

// Network to join. Leave the SSID empty to skip it: the board then starts its own
// access point (fixed IP 192.168.4.1).
const char* WIFI_SSID = "";
const char* WIFI_PASS = "";

// Name on the network (reachable as <HOSTNAME>.local) and of the board's own access point.
const char* HOSTNAME = "aircompress-valve";
const char* AP_SSID = "aircompress-valve";
const char* AP_PASS = "change-this-password";  // at least 8 characters
