<template>
  <v-navigation-drawer
    :mini-variant="miniDrawer"
    v-model="showDrawer"
    fixed
    app
  >
    <v-list>
      <template>
        <div v-for="item in items" :key="item.title">
          <v-list-group v-if="item.items" v-model="item.active" no-action>
            <template v-slot:activator>
                <v-list-item-icon>
                  <v-icon v-text="item.action"></v-icon>
                </v-list-item-icon>
                <v-list-item-content>
                  <v-list-item-title v-text="item.title"></v-list-item-title>
                </v-list-item-content>
            </template>

            <v-list-item
              v-for="child in item.items"
              :key="child.title"
              :to="child.to"
            >
              <v-list-item-icon>
                <v-icon v-text="child.action"></v-icon>
              </v-list-item-icon>
              <v-list-item-content>
                <v-list-item-title v-text="child.title"></v-list-item-title>
              </v-list-item-content>
            </v-list-item>
          </v-list-group>
          <v-list-item v-else :to="item.to" link>
            <v-list-item-icon>
              <v-icon v-text="item.action"></v-icon>
            </v-list-item-icon>
            <v-list-item-content>
              <v-list-item-title v-text="item.title"></v-list-item-title>
            </v-list-item-content>
          </v-list-item>
        </div>
      </template>
    </v-list>
    <template v-slot:append>
      <v-list>
        <v-list-item @click="logout">
          <v-list-item-action>
            <v-icon>close</v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>Logout</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
        <v-divider></v-divider>
        <v-list-item @click="switchMiniDrawer">
          <v-list-item-action>
            <v-icon
              v-html="miniDrawer ? 'chevron_right' : 'chevron_left'"
            ></v-icon>
          </v-list-item-action>
          <v-list-item-content>
            <v-list-item-title>Collapse</v-list-item-title>
          </v-list-item-content>
        </v-list-item>
      </v-list>
    </template>
  </v-navigation-drawer>
</template>

<script lang="ts">
import { Vue, Component } from 'vue-property-decorator';
import {
  readDashboardMiniDrawer,
  readDashboardShowDrawer,
  readHasAdminAccess,
} from '@/store/main/getters';
import {
  commitSetDashboardShowDrawer,
  commitSetDashboardMiniDrawer,
} from '@/store/main/mutations';
import { dispatchUserLogOut } from '@/store/main/actions';

@Component
export default class NavigationDrawer extends Vue {
  public items = [
    {
      action: 'web',
      title: 'Dashboard',
      to: '/main/dashboard',
    },
    {
      action: 'biotech',
      title: 'Projects',
      to: '/main/projects/all',
    },
    {
      action: 'insights',
      title: 'Models',
      to: '/main/models/all',
    },
    {
      action: 'topic',
      title: 'Datasets',
      to: '/main/datasets/all',
    },
    {
      action: 'gesture',
      title: 'Annotations',
      to: '/main/annotations/all',
    },
    {
      action: 'select_all',
      title: 'Segmentations',
      to: '/main/segmentations/all',
    },
    {
      action: 'admin_panel_settings',
      admin_only: true,
      items: [
        { action: 'group', title: 'Manage Users', to: '/main/admin/users/all' },
        {
          action: 'person_add',
          title: 'Create User',
          to: '/main/admin/users/create',
        },
      ],
      title: 'Admin',
    },
    {
      action: 'settings',
      items: [
        { action: 'person', title: 'Profile', to: '/main/profile/view' },
        { action: 'edit', title: 'Edit Profile', to: '/main/profile/edit' },
        {
          action: 'vpn_key',
          title: 'Change Password',
          to: '/main/profile/password',
        },
      ],
      title: 'Settings',
    },
  ];

  get miniDrawer() {
    return readDashboardMiniDrawer(this.$store);
  }

  get showDrawer() {
    return readDashboardShowDrawer(this.$store);
  }

  set showDrawer(value) {
    commitSetDashboardShowDrawer(this.$store, value);
  }

  public switchMiniDrawer() {
    commitSetDashboardMiniDrawer(
      this.$store,
      !readDashboardMiniDrawer(this.$store),
    );
  }

  public async logout() {
    await dispatchUserLogOut(this.$store);
  }

  public get hasAdminAccess() {
    return readHasAdminAccess(this.$store);
  }
}
</script>

<style>
</style>