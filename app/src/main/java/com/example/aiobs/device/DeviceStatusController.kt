package com.example.aiobs.device

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiInfo
import android.os.BatteryManager
import android.os.Build
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * V3 设备状态读取层，保持 minSdk 24 兼容。
 */
class DeviceStatusController(
    private val context: Context
) {

    data class Snapshot(
        val batteryPercent: Int,
        val temperatureC: Float,
        val network: String
    )

    private val appContext = context.applicationContext

    private val connectivityManager: ConnectivityManager by lazy {
        appContext.getSystemService(
            Context.CONNECTIVITY_SERVICE
        ) as ConnectivityManager
    }

    fun snapshot(): Snapshot {
        val battery = getBatteryAndTemp()

        return Snapshot(
            batteryPercent = battery.first,
            temperatureC = battery.second,
            network = getNetworkInfo()
        )
    }

    private fun getBatteryAndTemp(): Pair<Int, Float> {
        val intent = appContext.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )

        val level = intent?.getIntExtra(
            BatteryManager.EXTRA_LEVEL,
            -1
        ) ?: -1

        val temperatureRaw = intent?.getIntExtra(
            BatteryManager.EXTRA_TEMPERATURE,
            0
        ) ?: 0

        return level to (temperatureRaw / 10f)
    }

    private fun getNetworkInfo(): String {
        return try {
            val network = connectivityManager.activeNetwork
                ?: return "无网络"

            val capabilities = connectivityManager
                .getNetworkCapabilities(network)
                ?: return "无网络"

            when {
                capabilities.hasTransport(
                    NetworkCapabilities.TRANSPORT_WIFI
                ) -> wifiInfo(capabilities)

                capabilities.hasTransport(
                    NetworkCapabilities.TRANSPORT_CELLULAR
                ) -> cellularInfo()

                capabilities.hasTransport(
                    NetworkCapabilities.TRANSPORT_ETHERNET
                ) -> "以太网"

                else -> "未知网络"
            }
        } catch (e: SecurityException) {
            Log.w(TAG, "Network permission denied", e)
            "网络权限不足"
        } catch (e: Exception) {
            Log.w(TAG, "Network status failed", e)
            "网络状态未知"
        }
    }

    private fun wifiInfo(
        capabilities: NetworkCapabilities
    ): String {
        // transportInfo 可用于 API 29+；minSdk 24 保留兼容分支。
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            return "WIFI"
        }

        val info = capabilities.transportInfo as? WifiInfo
            ?: return "WIFI"

        val dbm = info.rssi
        val level = calculateWifiSignalLevel(dbm)

        return "WIFI (${signalText(level)}, ${dbm}dBm)"
    }

    /** 不调用已 deprecated 的 WifiManager.calculateSignalLevel。 */
    private fun calculateWifiSignalLevel(dbm: Int): Int {
        return when {
            dbm >= -55 -> 4
            dbm >= -65 -> 3
            dbm >= -75 -> 2
            dbm >= -85 -> 1
            else -> 0
        }
    }

    private fun cellularInfo(): String {
        val manager = appContext.getSystemService(
            Context.TELEPHONY_SERVICE
        ) as TelephonyManager

        // TelephonyManager.signalStrength 要求 API 28+。
        val signal = if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.P
        ) {
            try {
                manager.signalStrength
            } catch (e: SecurityException) {
                Log.w(
                    TAG,
                    "Unable to read cellular signal strength",
                    e
                )
                null
            }
        } else {
            null
        }

        val level = signal?.level ?: 0
        var dbmText = ""

        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(
                appContext,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            val exactDbm = signal
                ?.cellSignalStrengths
                ?.firstOrNull()
                ?.dbm

            if (
                exactDbm != null &&
                exactDbm != Int.MAX_VALUE
            ) {
                dbmText = ", ${exactDbm}dBm"
            }
        }

        return "移动数据 (${signalText(level)}$dbmText)"
    }

    private fun signalText(level: Int): String {
        return when {
            level >= 4 -> "满格"
            level == 3 -> "良好"
            level == 2 -> "一般"
            level == 1 -> "较弱"
            else -> "微弱"
        }
    }

    private companion object {
        const val TAG = "DeviceStatus"
    }
}
