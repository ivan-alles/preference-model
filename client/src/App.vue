<template>
  <div id="app">
    <div id="nav">
      <router-link to="/">Home</router-link> |
      <router-link to="/about">About</router-link>
    </div>
    <keep-alive>
        <router-view></router-view>
    </keep-alive>
  </div>
</template>

<style>
</style>
