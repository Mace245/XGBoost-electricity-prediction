#include <Arduino.h>
#include <PZEM004Tv30.h>
#include <AntaresESPMQTT.h>
#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <time.h>
#include <EEPROM.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// WiFi dan Antares Configuration
#define ACCESSKEY "5cd4cda046471a89:75f9e1c6b34bf41a"
#define WIFISSID "K206"
#define PASSWORD "kamar206"

#define projectName "UjiCoba_TA"
#define deviceName "TA_DKT1"

// EEPROM configuration
#define EEPROM_SIZE 16
#define DAILY_ENERGY_ADDR 0
#define TOTAL_ENERGY_ADDR 8

// Pin untuk ESP32
#define PZEM_RX_PIN 16
#define PZEM_TX_PIN 17
#define PZEM_SERIAL Serial2
#define CONSOLE_SERIAL Serial
#define RELAY_PIN 18

// Status relay
#define RELAY_ON HIGH
#define RELAY_OFF LOW

// Konstanta untuk pengaturan energi
#define DEFAULT_ENERGY_LIMIT 5.0  // Default energi limit dalam kWh
#define WARNING_THRESHOLD_80 0.8  // 80% dari batas energi
#define WARNING_THRESHOLD_90 0.9  // 90% dari batas energi
#define WIFI_RECONNECT_INTERVAL 30000  // Interval reconnect WiFi dalam ms

// Inisialisasi LCD I2C
#define LCD_ADDRESS 0x27
#define LCD_COLUMNS 20
#define LCD_ROWS 4
LiquidCrystal_I2C lcd(0x27, 20, 4);

AntaresESPMQTT antares(ACCESSKEY);
PZEM004Tv30 pzem(PZEM_SERIAL, PZEM_RX_PIN, PZEM_TX_PIN);

// NTP Client
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org");

// Variabel untuk pengukuran energi
float previousEnergyReading = 0.0;  // Pembacaan energi sebelumnya
float totalEnergy = 0.0;           // Total energi kumulatif
float dailyEnergy = 0.0;           // Total energi harian
float energyLimit = DEFAULT_ENERGY_LIMIT;  // Batas energi default
float voltage, current, power, energy;
int lastDay = -1;                  // Variabel untuk menyimpan hari terakhir
bool firstReading = true;          // Flag untuk pembacaan pertama
float energyLimit2 = DEFAULT_ENERGY_LIMIT;  //Variabel untuk mengirim kondisi batas limit pada mobile apps 

// Variabel untuk timing dan error handling
unsigned long previousPushMillis = 0;     // Timestamp untuk push data Antares
unsigned long previousReadMillis = 0;     // Timestamp untuk baca sensor
unsigned long previousWifiCheckMillis = 0; // Timestamp untuk cek status WiFi
const long PUSH_INTERVAL = 30000;         // Interval push data (30 detik)
const long READ_INTERVAL = 5000;          // Interval baca sensor (5 detik)
int sensorErrorCount = 0;                 // Counter error sensor
const int MAX_SENSOR_ERRORS = 5;          // Jumlah maksimum error berturut-turut

// Callback untuk menerima data dari Antares
void callback_antares(char topic[], byte payload[], unsigned int length) {
  antares.get(topic, payload, length);
  
  // Membaca limitEnergy dari Antares jika tersedia
  float receivedLimit = antares.getFloat("limitEnergy"); // float awalnya dihilangkan
  if (receivedLimit > 0) {
    energyLimit = receivedLimit;
    CONSOLE_SERIAL.print("Energy limit diperbarui: ");
    CONSOLE_SERIAL.println(energyLimit);
  }
  float receivedLimit2 = antares.getFloat("limitEnergy"); // float awalnya dihilangkan
  if (receivedLimit > 0) {
    energyLimit2 = receivedLimit;
    CONSOLE_SERIAL.print("Energy limit2 diperbarui: ");
    CONSOLE_SERIAL.println(energyLimit2);
  }
}

// Fungsi untuk push data ke Antares
void push_antares() {
  // Mengirim data ke Antares
  antares.add("Voltage", voltage);
  antares.add("Current", current);
  antares.add("Power", power);
  antares.add("Energy", energy);
  antares.add("TotalEnergy", isnan(totalEnergy) ? 0.0 : totalEnergy);
  antares.add("DailyEnergy", isnan(dailyEnergy) ? 0.0 : dailyEnergy);
  antares.add("energyLimit2", energyLimit2);
  antares.add("energyLimit90", WARNING_THRESHOLD_90 * energyLimit);
  antares.add("energyLimit80", WARNING_THRESHOLD_80 * energyLimit);
  
  // Publish data tanpa mencoba mengambil nilai return
  antares.publish(projectName, deviceName);
  CONSOLE_SERIAL.println("Data berhasil dipublikasikan ke Antares");
}

// Fungsi untuk mengontrol relay berdasarkan penggunaan energi
void relay_control() {
  if (dailyEnergy >= energyLimit) {
    digitalWrite(RELAY_PIN, RELAY_OFF);  // Matikan relay jika melebihi batas
    CONSOLE_SERIAL.println("RELAY: OFF (Batas energi terlampaui)");
  } else {
    digitalWrite(RELAY_PIN, RELAY_ON);   // Nyalakan relay jika masih dalam batas
    CONSOLE_SERIAL.println("RELAY: ON");
  }
}

// Fungsi untuk menampilkan peringatan penggunaan energi
void check_energy_warnings() {
  if (dailyEnergy >= WARNING_THRESHOLD_90 * energyLimit) {
    CONSOLE_SERIAL.println("PERINGATAN KRITIS: Penggunaan energi harian telah mencapai 90% dari limit!");
  } else if (dailyEnergy >= WARNING_THRESHOLD_80 * energyLimit) {
    CONSOLE_SERIAL.println("PERINGATAN: Penggunaan energi harian telah mencapai 80% dari limit!");
  }
}

// Fungsi untuk koneksi WiFi
bool connect_wifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }
  
  WiFi.begin(WIFISSID, PASSWORD);
  CONSOLE_SERIAL.print("Menghubungkan ke WiFi ");
  CONSOLE_SERIAL.println(WIFISSID);

  int timeout = 20; // Timeout dalam detik
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(500);
    CONSOLE_SERIAL.print(".");
    timeout--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    CONSOLE_SERIAL.println("\nWiFi Terhubung!");
    CONSOLE_SERIAL.print("IP address: ");
    CONSOLE_SERIAL.println(WiFi.localIP());
    return true;
  } else {
    CONSOLE_SERIAL.println("\nGagal terhubung ke WiFi!");
    return false;
  }
}

// Fungsi untuk menyimpan data energi ke EEPROM
void save_energy_data() {
  EEPROM.writeFloat(DAILY_ENERGY_ADDR, dailyEnergy);
  EEPROM.writeFloat(TOTAL_ENERGY_ADDR, totalEnergy);
  EEPROM.commit();
  CONSOLE_SERIAL.println("Data energi disimpan ke EEPROM");
}

// Fungsi untuk memuat data energi dari EEPROM
void load_energy_data() {
  dailyEnergy = EEPROM.readFloat(DAILY_ENERGY_ADDR);
  totalEnergy = EEPROM.readFloat(TOTAL_ENERGY_ADDR);
  
  // Validate and set default values if invalid
  if (isnan(dailyEnergy) || dailyEnergy < 0) {
    dailyEnergy = 0.0;
    CONSOLE_SERIAL.println("Daily Energy invalid, reset to 0");
  }
  
  if (isnan(totalEnergy) || totalEnergy < 0) {
    totalEnergy = 0.0;
    CONSOLE_SERIAL.println("Total Energy invalid, reset to 0");
  }
  
  CONSOLE_SERIAL.println("Data energi dimuat dari EEPROM:");
  CONSOLE_SERIAL.print("Daily Energy: "); CONSOLE_SERIAL.println(dailyEnergy);
  CONSOLE_SERIAL.print("Total Energy: "); CONSOLE_SERIAL.println(totalEnergy);
}

// Fungsi untuk membaca data dari sensor PZEM
bool read_sensor_data() {
  voltage = pzem.voltage();
  current = pzem.current();
  power = pzem.power();
  energy = pzem.energy();

  // Cek apakah data valid
  if (isnan(voltage) || isnan(current) || isnan(power) || isnan(energy)) {
    sensorErrorCount++;
    CONSOLE_SERIAL.print("Gagal membaca data dari PZEM004T! (");
    CONSOLE_SERIAL.print(sensorErrorCount);
    CONSOLE_SERIAL.println(" kali berturut-turut)");
    
    if (sensorErrorCount >= MAX_SENSOR_ERRORS) {
      CONSOLE_SERIAL.println("ERROR: Terlalu banyak kegagalan pembacaan sensor!");
    }
    return false;
  }

  // Reset error count jika pembacaan berhasil
  sensorErrorCount = 0;
  
  // Jika ini pembacaan pertama, simpan sebagai nilai awal
  if (firstReading) {
    previousEnergyReading = energy;
    firstReading = false;
    return true;
  }
  
  // Hitung selisih energi (hanya jika nilai energi baru > nilai sebelumnya)
  if (energy >= previousEnergyReading) {
    float energyDelta = energy - previousEnergyReading;
    // Update total dan daily energy hanya jika delta positif dan wajar
    if (energyDelta > 0 && energyDelta < 1.0) {  // Batasi delta energi yang masuk akal
      totalEnergy += energyDelta;
      dailyEnergy += energyDelta;
      
      // Simpan data ke EEPROM setiap ada perubahan energi
      save_energy_data();
    }
  }
  
  // Update nilai energi terakhir
  previousEnergyReading = energy;
  return true;
}

// Fungsi untuk menampilkan data sensor pada LCD
void display_sensor_data_lcd() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("V: ");
  lcd.print(voltage);
  lcd.print(" V");

  lcd.setCursor(0, 1);
  lcd.print("I: ");
  lcd.print(current);
  lcd.print(" A");

  lcd.setCursor(0, 2);
  lcd.print("P: ");
  lcd.print(power);
  lcd.print(" W");

  lcd.setCursor(0, 3);
  lcd.print("Etd: ");
  lcd.print(dailyEnergy, 3);
  lcd.print(" kWh");
}

void display_sensor_data() {
  // Mendapatkan waktu saat ini dari NTP
  if (getLocalTime(&timeinfo)){
    timeClient.update();
  }
  time_t epochTime = timeClient.getEpochTime();
  struct tm *timeInfo = localtime(&epochTime);
  int currentDay = timeInfo->tm_mday;
  int currentMonth = timeInfo->tm_mon + 1;
  int currentYear = timeInfo->tm_year + 1900;
  int currentHour = timeInfo->tm_hour;
  int currentMinute = timeInfo->tm_min;
  int currentSecond = timeInfo->tm_sec;

  // Menampilkan data di Serial Monitor
  CONSOLE_SERIAL.println("\n===== DATA PENGUKURAN DAYA =====");
  CONSOLE_SERIAL.print("Voltage: "); CONSOLE_SERIAL.print(voltage); CONSOLE_SERIAL.println(" V");
  CONSOLE_SERIAL.print("Current: "); CONSOLE_SERIAL.print(current); CONSOLE_SERIAL.println(" A");
  CONSOLE_SERIAL.print("Power: "); CONSOLE_SERIAL.print(power); CONSOLE_SERIAL.println(" W");
  CONSOLE_SERIAL.print("Energy: "); CONSOLE_SERIAL.print(energy, 3); CONSOLE_SERIAL.println(" kWh");
  CONSOLE_SERIAL.print("Total Energy: "); CONSOLE_SERIAL.print(totalEnergy, 3); CONSOLE_SERIAL.println(" kWh");
  CONSOLE_SERIAL.print("Daily Energy: "); CONSOLE_SERIAL.print(dailyEnergy, 3); CONSOLE_SERIAL.println(" kWh");
  CONSOLE_SERIAL.print("Energy Limit: "); CONSOLE_SERIAL.print(energyLimit, 3); CONSOLE_SERIAL.println(" kWh");
  CONSOLE_SERIAL.printf("Time: %02d:%02d:%02d\n", currentHour, currentMinute, currentSecond);
  CONSOLE_SERIAL.printf("Date: %02d/%02d/%d\n", currentDay, currentMonth, currentYear);
  CONSOLE_SERIAL.println("==================================");

  // Cek apakah hari berganti
  if (lastDay != -1 && currentDay != lastDay) {
    // dailyEnergy = 0.0;  // Reset energi harian
    CONSOLE_SERIAL.println("Hari berganti, dailyEnergy direset menjadi 0.");
    save_energy_data();
  }
  lastDay = currentDay;

  // Tampilkan data pada LCD
  display_sensor_data_lcd();
}

void setup() {
  // Debugging Serial
  CONSOLE_SERIAL.begin(115200);
  CONSOLE_SERIAL.println("\nSistem Monitoring Energi dengan ESP32 dan PZEM004T");
  
  // Inisialisasi EEPROM
  EEPROM.begin(EEPROM_SIZE);
  
  // Muat data energi dari EEPROM
  load_energy_data();
  
  // Set pin relay sebagai OUTPUT dan matikan saat awal
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, RELAY_OFF);
  CONSOLE_SERIAL.println("Relay diinisialisasi dan dimatikan");
  
  // Koneksi ke WiFi
  if (connect_wifi()) {
    // Koneksi ke Antares MQTT jika WiFi terhubung
    antares.setDebug(true);
    antares.setMqttServer();
    antares.setCallback(callback_antares);
    CONSOLE_SERIAL.println("Setup Antares MQTT selesai");
  }

  // Inisialisasi NTP Client
  timeClient.begin();
  timeClient.setTimeOffset(25200); // GMT+7 (WIB)
  CONSOLE_SERIAL.println("NTP Client diinisialisasi");

  // Inisialisasi LCD I2C
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Sistem Monitoring");
  lcd.setCursor(0, 1);
  lcd.print("Energi ESP32");
  delay(2000);
  
  CONSOLE_SERIAL.println("Setup selesai!");
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Periksa koneksi WiFi secara periodik
  if (currentMillis - previousWifiCheckMillis >= WIFI_RECONNECT_INTERVAL) {
    previousWifiCheckMillis = currentMillis;
    if (WiFi.status() != WL_CONNECTED) {
      CONSOLE_SERIAL.println("WiFi terputus, mencoba menghubungkan kembali...");
      connect_wifi();
    }
  }
  
  // Pastikan tetap terhubung ke MQTT Server jika WiFi terhubung
  if (WiFi.status() == WL_CONNECTED) {
    antares.checkMqttConnection();
  }
  
  // Ambil data terakhir dari Antares secara periodik
  if (WiFi.status() == WL_CONNECTED && currentMillis - previousPushMillis >= PUSH_INTERVAL) {
    previousPushMillis = currentMillis;
    
    // Ambil data terakhir dari Antares (termasuk energyLimit)
    antares.retrieveLastData(projectName, deviceName);
    
    // Push data ke Antares
    push_antares();
  }
  
  // Baca data sensor secara periodik
  if (currentMillis - previousReadMillis >= READ_INTERVAL) {
    previousReadMillis = currentMillis;
    
    // Baca data sensor
    if (read_sensor_data()) {
      // Tampilkan data jika pembacaan berhasil
      display_sensor_data();
      
      // Periksa peringatan penggunaan energi
      check_energy_warnings();
      
      // Kontrol relay berdasarkan penggunaan energi
      relay_control();
    }
  }
}