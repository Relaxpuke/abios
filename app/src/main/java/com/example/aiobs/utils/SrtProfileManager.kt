package com.example.aiobs.utils

import android.content.Context
import android.content.SharedPreferences

object SrtProfileManager {
    private const val PREFS_NAME = "AIOBS_SRT_PREFS"
    private const val KEY_HOST = "srt_host"
    private const val KEY_PORT = "srt_port"
    private const val KEY_STREAM = "srt_stream"

    private fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getHost(context: Context): String = prefs(context)
        .getString(KEY_HOST, "100.74.255.34")
        .orEmpty()
        .ifBlank { "100.74.255.34" }

    fun getPort(context: Context): String = prefs(context)
        .getString(KEY_PORT, "8890")
        .orEmpty()
        .ifBlank { "8890" }

    fun getStreamName(context: Context): String = prefs(context)
        .getString(KEY_STREAM, "live")
        .orEmpty()
        .ifBlank { "live" }

    fun saveProfile(context: Context, host: String, port: String, stream: String) {
        prefs(context).edit()
            .putString(KEY_HOST, host.trim())
            .putString(KEY_PORT, port.trim())
            .putString(KEY_STREAM, stream.trim())
            .apply()
    }

    fun getFullUrl(context: Context): String {
        val host = getHost(context)
        val port = getPort(context)
        val stream = getStreamName(context).trimStart('/')
        return "srt://$host:$port/$stream?latency=3000"
    }
}
