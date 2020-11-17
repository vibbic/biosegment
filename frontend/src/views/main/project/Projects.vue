<template>
  <div>
    <v-toolbar light>
      <v-toolbar-title>
        Manage Projects
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn color="primary" to="/main/projects/create">Create Project</v-btn>
    </v-toolbar>
    <v-data-table :headers="headers" :items="projects" item-key="name">
      <template v-slot:item="{ item }">
        <tr>
          <td>{{ item.title }}</td>
          <td>
            <v-btn text :to="{name: 'main-projects-edit', params: {id: item.id}}">
              <v-icon>edit</v-icon>
            </v-btn>
            <v-btn text @click="deleteProject(item.id)">
              <v-icon>delete</v-icon>
            </v-btn>
          </td>
        </tr>
      </template>
    </v-data-table>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Store } from 'vuex';
import { Project } from '@/api';
import { readProjects } from '@/store/project/getters';
import { dispatchGetProjects, dispatchDeleteProject } from '@/store/project/actions';

@Component
export default class ProjectProjects extends Vue {
  public headers = [
    {
      text: 'Title',
      sortable: true,
      value: 'title',
      align: 'left',
    },
    {
      text: 'Actions',
      value: 'id',
    },
  ];
  get projects() {
    return readProjects(this.$store);
  }

  public async mounted() {
    await dispatchGetProjects(this.$store);
  }

  public async deleteProject(id: number) {
    await dispatchDeleteProject(this.$store, {id});
  }
}
</script>
