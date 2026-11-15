// Vacuum valve of the suction gripper.
// Adafruit Feather ESP32 V2 -> dual MOSFET module -> 24 V solenoid valve.
// Control over USB serial (115200 baud) or Wi-Fi HTTP:
//   Serial lines: ON / OFF / PULSE <ms> / STATUS     reply: "OK ON", "OK OFF" or "ERR ..."
//   HTTP:         /on  /off  /pulse?ms=500  /status
// The PC side is Arm/rfp_arm/valve.py; bench tool: Arm/scripts/gripper_bench.py.
//
// Wi-Fi names and passwords are not in this file: copy secrets.example.h to secrets.h
// (ignored by Git) and fill it in.

#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include "secrets.h"

// ESP32 pin wired to the MOSFET module's J1 signal pad (PWM/TRIG).
// Wired to A0 (GPIO26); GND is the pin next to it.
const int VALVE_PIN = 26;
const int LED_PIN = LED_BUILTIN;  // Onboard red #13 LED, mirrors the valve state

WebServer server(80);
bool valveOn = false;
unsigned long pulseStart = 0;
unsigned long pulseMs = 0;  // 0 = no pulse running

void setValve(bool on) {
  valveOn = on;
  pulseMs = 0;
  digitalWrite(VALVE_PIN, on ? HIGH : LOW);
  digitalWrite(LED_PIN, on ? HIGH : LOW);
}

void startPulse(unsigned long ms) {
  setValve(true);
  pulseStart = millis();
  pulseMs = ms;
}

String status() {
  return valveOn ? "ON" : "OFF";
}

// Command parser shared by serial and HTTP; returns a one-line reply
String handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd == "ON") {
    setValve(true);
  } else if (cmd == "OFF") {
    setValve(false);
  } else if (cmd.startsWith("PULSE")) {
    long ms = cmd.substring(5).toInt();
    if (ms <= 0) return "ERR PULSE needs ms > 0";
    startPulse(ms);
  } else if (cmd != "STATUS") {
    return "ERR unknown command: " + cmd;
  }
  return "OK " + status();
}

// With an SSID in secrets.h, join that network; if it is empty or the connection fails,
// start an access point (fixed IP 192.168.4.1).
void setupWifi() {
  if (strlen(WIFI_SSID) > 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("Connecting to WiFi");
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
      delay(250);
      Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("WiFi IP: " + WiFi.localIP().toString());
      if (MDNS.begin(HOSTNAME)) Serial.println("mDNS: " + String(HOSTNAME) + ".local");
      return;
    }
    Serial.println("WiFi connect failed, falling back to AP mode");
  }
  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  Serial.println("AP " + String(AP_SSID) + " IP: " + WiFi.softAPIP().toString());
}

void setupServer() {
  server.on("/on", [] { server.send(200, "text/plain", handleCommand("ON")); });
  server.on("/off", [] { server.send(200, "text/plain", handleCommand("OFF")); });
  server.on("/status", [] { server.send(200, "text/plain", handleCommand("STATUS")); });
  server.on("/pulse", [] {
    server.send(200, "text/plain", handleCommand("PULSE " + server.arg("ms")));
  });
  server.begin();
}

void setup() {
  // Drive the valve pin low first so the valve stays closed on power-up/reset
  pinMode(VALVE_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  setValve(false);

  Serial.begin(115200);
  delay(200);
  setupWifi();
  setupServer();
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    Serial.println(handleCommand(Serial.readStringUntil('\n')));
  }
  server.handleClient();

  if (pulseMs && millis() - pulseStart >= pulseMs) {
    setValve(false);
  }
}
