<template>
  <v-form v-model="valid" ref="form" lazy-validation>
    <v-text-field label="Title" v-model="project.title" required></v-text-field>
    <v-text-field
      label="Description"
      v-model="project.description"
    ></v-text-field>
    <v-menu
        v-model="start_dateMenu"
        :close-on-content-click="false"
        :nudge-right="40"
        transition="scale-transition"
        offset-y
        min-width="290px"
      >
        <template v-slot:activator="{ on, attrs }">
          <v-text-field
            v-model="project.start_date"
            label="Project Start Date"
            prepend-icon="mdi-calendar"
            readonly
            v-bind="attrs"
            v-on="on"
          ></v-text-field>
        </template>
        <v-date-picker
          v-model="project.start_date"
          @input="start_dateMenu = false"
        ></v-date-picker>
      </v-menu>
    <v-menu
        v-model="stop_dateMenu"
        :close-on-content-click="false"
        :nudge-right="40"
        transition="scale-transition"
        offset-y
        min-width="290px"
      >
        <template v-slot:activator="{ on, attrs }">
          <v-text-field
            v-model="project.stop_date"
            label="Project Stop Date"
            prepend-icon="mdi-calendar"
            readonly
            v-bind="attrs"
            v-on="on"
          ></v-text-field>
        </template>
        <v-date-picker
          v-model="project.stop_date"
          @input="stop_dateMenu = false"
        ></v-date-picker>
      </v-menu>
  </v-form>
</template>

<script lang="ts">
import { Component, Model, Vue } from 'vue-property-decorator';
import { ProjectCreate } from '@/api';
import { defaultProject } from '@/interfaces';

@Component
export default class ProjectForm extends Vue {
  @Model('project', { type: ProjectCreate }) public project!: ProjectCreate;
  public valid = false;
  public start_dateMenu = false
  public stop_dateMenu = false
}
</script>
